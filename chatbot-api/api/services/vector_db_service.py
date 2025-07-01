# services/vector_db_service.py
import chromadb
import logging
import os

logger = logging.getLogger(__name__)

# --- Configuration ---
# Directory where ChromaDB will store its persistent data files.
# Make sure this directory exists or ChromaDB has permissions to create it.
CHROMA_DATA_PATH = "chroma_persistent_data"
# Name for the collection within ChromaDB (like a table name)
COLLECTION_NAME = "document_chunks_prod" # Choose a descriptive name

# Ensure the data directory exists
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

# --- Initialize ChromaDB Client (Persistent) ---
try:
    # Create a persistent client that saves data to the specified path
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    logger.info(f"ChromaDB PersistentClient initialized. Data path: {CHROMA_DATA_PATH}")

    # --- Get or Create the Collection ---
    # You should specify the distance function if it matters (default is L2 for PersistentClient usually, but good to be explicit)
    # Match this to your embedding model and previous setup (L2 for IndexFlatL2)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"} # Specify L2 distance (Euclidean)
        # embedding_function=None # We provide embeddings manually
    )
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' accessed/created successfully.")

except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
    # If the vector DB fails to load, the app likely can't function.
    raise RuntimeError("Could not initialize persistent ChromaDB") from e

# --- FastAPI Dependency ---
# This function will be used by FastAPI's Depends() to inject the collection
# into endpoint functions.
def get_chroma_collection() -> chromadb.Collection:
    """FastAPI dependency function to get the ChromaDB collection."""
    # In a real application, you might add error handling or re-initialization logic here if needed,
    # but for this setup, we rely on the initial client/collection creation succeeding.
    if collection is None:
         logger.error("ChromaDB collection is not available!")
         raise RuntimeError("ChromaDB collection not initialized")
    return collection
