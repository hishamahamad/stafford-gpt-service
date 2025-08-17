# Model configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# Chat model configuration  
CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Default namespace values
DEFAULT_NAMESPACE = "customer"
INTERNAL_NAMESPACE = "internal"

# Scraping configuration
SELECTORS_TO_REMOVE = ["header", "nav", "footer", ".cookie-banner", "#cookie-consent", ".popup"]
SCRAPE_TIMEOUT = 30000
NETWORK_IDLE_TIMEOUT = 15000
CONTENT_WAIT_TIME = 3000

# Retry configuration
MAX_EMBED_RETRIES = 5

# Query limits
DEFAULT_LIMIT = 100
DEFAULT_OFFSET = 0
VECTOR_SEARCH_LIMIT = 10
MAX_CHUNKS_PER_SOURCE = 3