import os
import requests
import asyncio
from typing import List
from lxml import etree
from fastapi import HTTPException
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from playwright.async_api import async_playwright
from backoff import on_exception, expo
from constants import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SELECTORS_TO_REMOVE,
    SCRAPE_TIMEOUT, NETWORK_IDLE_TIMEOUT, CONTENT_WAIT_TIME, MAX_EMBED_RETRIES
)


def get_embeddings():
    """Initialize OpenAI embeddings with retry logic"""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return on_exception(expo, Exception, max_tries=MAX_EMBED_RETRIES)(embeddings.embed_documents)


def get_text_splitter():
    """Initialize text splitter with configured chunk size and overlap"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )


async def scrape_with_playwright(url: str) -> str:
    """Custom playwright scraper that waits for content to stabilize"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # Go to URL
            await page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT)
            
            # Wait for network to be idle (no requests for 500ms)
            await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)
            
            # Additional wait for any remaining dynamic content
            await page.wait_for_timeout(CONTENT_WAIT_TIME)
            
            # Remove unwanted elements
            for selector in SELECTORS_TO_REMOVE:
                await page.evaluate(f"""
                    document.querySelectorAll('{selector}').forEach(el => el.remove());
                """)
            
            # Get the content
            content = await page.content()
            
            # Extract just the text content
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(el => el.remove());
                    
                    // Get clean text content
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            
            await browser.close()
            return text_content.strip()
            
        except Exception as e:
            await browser.close()
            raise Exception(f"Failed to scrape {url}: {str(e)}")


def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parse sitemap XML and extract all URLs."""
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = etree.fromstring(response.content)
        
        # Handle different sitemap formats
        urls = []
        
        # Standard sitemap with <url><loc> tags
        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        loc_elements = root.xpath('//sitemap:url/sitemap:loc/text()', namespaces=namespaces)
        urls.extend(loc_elements)
        
        # Sitemap index with <sitemap><loc> tags (recursive sitemaps)
        sitemap_elements = root.xpath('//sitemap:sitemap/sitemap:loc/text()', namespaces=namespaces)
        for sitemap_loc in sitemap_elements:
            try:
                nested_urls = parse_sitemap(sitemap_loc)
                urls.extend(nested_urls)
            except Exception as e:
                print(f"Failed to parse nested sitemap {sitemap_loc}: {e}")
                continue
        
        # Fallback: try without namespace for non-standard sitemaps
        if not urls:
            loc_elements = root.xpath('//url/loc/text()')
            urls.extend(loc_elements)
            
            sitemap_elements = root.xpath('//sitemap/loc/text()')
            for sitemap_loc in sitemap_elements:
                try:
                    nested_urls = parse_sitemap(sitemap_loc)
                    urls.extend(nested_urls)
                except Exception:
                    continue
        
        return list(set(urls))  # Remove duplicates
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse sitemap: {e}")


def setup_database_tables(cursor):
    """Create database tables if they don't exist"""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(1536),
        source TEXT,
        doc_type TEXT,
        namespace TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id TEXT REFERENCES sessions(id),
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """)


def hybrid_search(cursor, query: str, namespace: str, query_embedding, limit: int = 10):
    """Perform hybrid search with full-text and vector similarity"""
    # Hybrid retrieval: full-text filter + vector ranking
    sql_hybrid = """
    SELECT content, source
      FROM documents
     WHERE to_tsvector('english', content)
           @@ plainto_tsquery('english', %s)
       AND (%s IS NULL OR namespace = %s)
     ORDER BY embedding <-> %s::vector
     LIMIT %s
    """
    params_hybrid = [query, namespace, namespace, query_embedding, limit]
    cursor.execute(sql_hybrid, params_hybrid)
    rows = cursor.fetchall()

    # Fallback: pure vector search if no full-text hits
    if not rows:
        if namespace:
            sql_vector = """
            SELECT content, source
              FROM documents
             WHERE namespace = %s
             ORDER BY embedding <-> %s::vector
             LIMIT %s
            """
            params_vector = [namespace, query_embedding, limit]
        else:
            sql_vector = """
            SELECT content, source
              FROM documents
             ORDER BY embedding <-> %s::vector
             LIMIT %s
            """
            params_vector = [query_embedding, limit]

        cursor.execute(sql_vector, params_vector)
        rows = cursor.fetchall()
    
    return rows


def group_context_by_source(rows, max_chunks_per_source=3):
    """Group retrieved context by source/university"""
    from collections import defaultdict
    from urllib.parse import urlparse
    
    source_groups = defaultdict(list)
    for text, src in rows:
        # Extract domain/university from source URL
        domain = urlparse(src).netloc if src.startswith('http') else src
        source_groups[domain].append(text)
    
    # Build grouped context blocks
    context_blocks = []
    for i, (source, texts) in enumerate(source_groups.items()):
        combined_text = "\n".join(texts[:max_chunks_per_source])
        context_blocks.append(f"[University/Source {i+1}: {source}]\n{combined_text}")
    
    return "\n\n".join(context_blocks)