import os
import uuid
import time
from typing import List, Optional
from fastapi import FastAPI, Form, Body, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from pgvector.psycopg2 import register_vector
import psycopg2
import anyio

from constants import (
    EMBEDDING_MODEL, CHAT_MODEL, TEMPERATURE, DEFAULT_NAMESPACE,
    VECTOR_SEARCH_LIMIT, MAX_CHUNKS_PER_SOURCE,
    SYSTEM_PROMPT
)
from utils import (
    get_embeddings, get_text_splitter, scrape_with_playwright, parse_sitemap,
    setup_database_tables, hybrid_search, group_context_by_source
)

load_dotenv()

app = FastAPI(title="RAG Service", description="Unified ingestion and query service")


def connect_to_database(max_retries=5, delay=5):
    """Connect to database with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to database (attempt {attempt + 1}/{max_retries})...")
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            register_vector(conn)
            print("✅ Database connected successfully!")
            return conn
        except psycopg2.OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("💥 Max retries reached. Database connection failed.")
                raise
        except Exception as e:
            print(f"💥 Unexpected error connecting to database: {e}")
            raise


# Database setup with retry logic
conn = connect_to_database()
cur = conn.cursor()
setup_database_tables(cur)
conn.commit()

# Initialize models
embeddings_instance = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

safe_embed = get_embeddings()
splitter = get_text_splitter()
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)

# this should ideally be a function "build_prompt(chat_history, context_blocks, question)"
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """
----- Conversation So Far -----
{chat_history}

----- Retrieved Context (Grouped by University/Source) -----
{context_blocks}

----- New User Question -----
{question}

RESPONSE FORMAT:
- If this is an exploratory question (e.g., "what business programs are available"), structure your response by university
- If this is a specific question about one university, focus only on that institution  
- Always make it clear which university offers which programs
- Never combine information from different universities into one recommendation
- Keep responses CONCISE and focused - don't repeat previously mentioned information

CRITICAL RULES:
- Present programs and universities based on the context provided
- Keep each university's offerings clearly separated - don't mix details from different institutions
- For "are there other..." questions: If only one option exists, give a brief "No" answer
- Be helpful but avoid verbose explanations when a simple answer suffices
"""
    )
])


# Pydantic models
class BulkURLRequest(BaseModel):
    urls: List[str]
    namespace: str = DEFAULT_NAMESPACE


class DocumentRequest(BaseModel):
    content: str
    source: str = "manual"
    doc_type: str = "text"
    namespace: str = DEFAULT_NAMESPACE


class BulkDocumentRequest(BaseModel):
    documents: List[DocumentRequest]


# Ingestion endpoints
@app.post("/ingest/urls")
async def ingest_bulk_urls(request: BulkURLRequest):
    """Ingest multiple URLs by scraping their content"""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    total_chunks = 0
    failed_urls = []
    successful_urls = []
    
    for url in request.urls:
        try:
            # Use custom scraper that handles dynamic content properly
            text_content = await scrape_with_playwright(url)
            
            print(f"DEBUG: URL {url} - Scraped {len(text_content)} characters")
            print(f"DEBUG: Content preview: {text_content[:200]}")
            
            if not text_content:
                failed_urls.append({"url": url, "error": "No content loaded"})
                continue
            
            # Create document object and split into chunks
            doc = Document(page_content=text_content, metadata={"source": url})
            docs = splitter.split_documents([doc])
            
            if not docs:
                failed_urls.append({"url": url, "error": "No chunks created"})
                continue
            
            # Embed chunks
            vectors = safe_embed([d.page_content for d in docs]) 
     
            # Store in database
            # TODO: store data in a data structure and do bulk update outside the loop, can be made a function and ideally
            # be used in ingest_documents as well
            for doc, vec in zip(docs, vectors):
                cur.execute(
                    "INSERT INTO documents (content, embedding, source, doc_type, namespace) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (doc.page_content, vec, url, "web", request.namespace)
                )
            
            total_chunks += len(docs)
            successful_urls.append(url)
            
        except Exception as e:
            failed_urls.append({"url": url, "error": str(e)})
            continue
    
    # Commit all changes
    conn.commit()
    
    return {
        "status": "completed",
        "total_urls_provided": len(request.urls),
        "successful_ingestions": len(successful_urls),
        "failed_ingestions": len(failed_urls),
        "total_chunks": total_chunks,
        "failed_urls": failed_urls
    }


@app.post("/ingest/documents")
async def ingest_documents(request: BulkDocumentRequest):
    """Ingest documents directly (not URLs)"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    total_chunks = 0
    failed_docs = []
    successful_docs = []
    
    for i, doc_req in enumerate(request.documents):
        try:
            if not doc_req.content.strip():
                failed_docs.append({"index": i, "error": "Empty content"})
                continue
            
            # Create document and split into chunks
            doc = Document(
                page_content=doc_req.content, 
                metadata={"source": doc_req.source}
            )
            docs = splitter.split_documents([doc])
            
            if not docs:
                failed_docs.append({"index": i, "error": "No chunks created"})
                continue
            
            # Embed chunks
            vectors = safe_embed([d.page_content for d in docs])
            
            # Store in database
            # TODO: store data in a data structure and do bulk update outside the loop
            for chunk_doc, vec in zip(docs, vectors):
                cur.execute(
                    "INSERT INTO documents (content, embedding, source, doc_type, namespace) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (chunk_doc.page_content, vec, doc_req.source, doc_req.doc_type, doc_req.namespace)
                )
            
            total_chunks += len(docs)
            successful_docs.append({"index": i, "source": doc_req.source})
            
        except Exception as e:
            failed_docs.append({"index": i, "error": str(e)})
            continue
    
    # Commit all changes
    conn.commit()
    
    return {
        "status": "completed",
        "total_documents_provided": len(request.documents),
        "successful_ingestions": len(successful_docs),
        "failed_ingestions": len(failed_docs),
        "total_chunks": total_chunks,
        "successful_documents": successful_docs,
        "failed_documents": failed_docs
    }


@app.post("/ingest/sitemap")
async def ingest_sitemap(
    sitemap_url: str = Form(..., description="URL of the sitemap to parse and ingest"),
    namespace: str = Form(default=DEFAULT_NAMESPACE, description="Namespace for the ingested documents")
):
    """Parse sitemap and ingest all URLs found"""
    try:
        urls = parse_sitemap(sitemap_url)
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs found in sitemap")
        
        # Use the bulk URL ingest endpoint
        request = BulkURLRequest(urls=urls, namespace=namespace)
        return await ingest_bulk_urls(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process sitemap: {str(e)}")


# Query endpoint
@app.post("/query")
def query(
    q: str = Body(..., embed=True, description="The user's latest question"),
    namespace: Optional[str] = Body(None, description="'internal' or 'customer'"),
    session_id: Optional[str] = Body(None, description="UUID of the chat session")
):
    """Query the knowledge base and get AI-generated response"""
    # 1) Ensure session exists
    if not session_id:
        session_id = str(uuid.uuid4())
        cur.execute("INSERT INTO sessions (id) VALUES (%s)", (session_id,))
        conn.commit()

    # 2) Log the user's message
    cur.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, "user", q),
    )
    conn.commit()

    # 3) Retrieve full chat history
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at",
        (session_id,),
    )
    chat_history = "\n".join(f"[{role.upper()}] {txt}" for role, txt in cur.fetchall())

    # 4) Embed the user question
    q_emb = embeddings_instance.embed_query(q)

    # 5) Perform hybrid search
    rows = hybrid_search(cur, q, namespace, q_emb, VECTOR_SEARCH_LIMIT)

    # 6) Group context by source/university
    context_blocks = group_context_by_source(rows, MAX_CHUNKS_PER_SOURCE)

    # 7) Generate the answer
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(
        chat_history=chat_history,
        context_blocks=context_blocks,
        question=q
    )

    # 8) Log assistant response
    cur.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, "assistant", answer),
    )
    conn.commit()

    return {"answer": answer, "session_id": session_id}


# Document management endpoints
# "/documents" with "get" returns a list and with "delete" deletes!? i'd preffer different paths just to be safe
# haven't been using REST lately so i might be wrong jzk
@app.get("/documents")
async def list_documents(
    namespace: str = None,
    limit: int = 100,
    offset: int = 0
):
    """List documents in the knowledge base"""
    query = """
    SELECT id, source, doc_type, namespace, created_at, 
           LEFT(content, 200) as content_preview
    FROM documents
    """
    params = []
    
    if namespace:
        query += " WHERE namespace = %s"
        params.append(namespace)
    
    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    cur.execute(query, params)
    results = cur.fetchall()
    
    documents = []
    for row in results:
        documents.append({
            "id": row[0],
            "source": row[1],
            "doc_type": row[2],
            "namespace": row[3],
            "created_at": row[4].isoformat() if row[4] else None,
            "content_preview": row[5]
        })
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM documents"
    count_params = []
    
    if namespace:
        count_query += " WHERE namespace = %s"
        count_params.append(namespace)
    
    cur.execute(count_query, count_params)
    total = cur.fetchone()[0]
    
    return {
        "documents": documents,
        "total": total,
        "limit": limit,
        "offset": offset
    }

# it only deletes one document so url is misleading
@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a specific document by ID"""
    cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    conn.commit()
    return {"status": "deleted", "document_id": document_id}

# url name and function name don't match
@app.delete("/documents")
async def delete_documents_by_filter(source: str = None, namespace: str = None):
    """Delete documents by source or namespace"""
    if not source and not namespace:
        raise HTTPException(status_code=400, detail="Must provide either source or namespace parameter")
    
    query = "DELETE FROM documents WHERE"
    params = []
    conditions = []
    
    if source:
        conditions.append(" source = %s")
        params.append(source)
    
    if namespace:
        conditions.append(" namespace = %s")
        params.append(namespace)
    
    query += " AND".join(conditions)
    
    cur.execute(query, params)
    deleted_count = cur.rowcount
    conn.commit()
    
    return {
        "status": "deleted",
        "deleted_count": deleted_count,
        "filter": {"source": source, "namespace": namespace}
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-unified"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)