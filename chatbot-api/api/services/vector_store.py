# services/vector_store.py (or wherever VectorStore is defined)

import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Any # Added Dict, Tuple, Any

logger = logging.getLogger(__name__)


class VectorStore:
    # Using dimension 384 as per your example. Make sure this matches your embedding model.
    def __init__(self, dimension: int = 384) -> None:
        self.index = faiss.IndexFlatL2(dimension)
        # Store metadata for each vector. The index in this list
        # corresponds to the index of the vector in the FAISS index.
        self.metadata: List[Dict[str, Any]] = []
        logger.info(f"Initialized VectorStore with dimension {dimension}")

    def add_chunks(self, chunk_embeddings: List[np.ndarray], chunk_metadatas: List[Dict[str, Any]]) -> None:
        """
        Adds multiple chunk embeddings and their corresponding metadata.

        Args:
            chunk_embeddings: A list of numpy arrays, each representing a chunk's embedding.
            chunk_metadatas: A list of dictionaries, each containing metadata for the
                             corresponding chunk (e.g., {'doc_id': 1, 'filename': 'a.pdf', 'text': 'chunk content...'}).
        """
        if len(chunk_embeddings) != len(chunk_metadatas):
            logger.error("Mismatch between number of embeddings and metadatas.")
            raise ValueError("Number of embeddings and metadatas must match.")

        if not chunk_embeddings:
            logger.warning("add_chunks called with empty lists.")
            return # Nothing to add

        processed_embeddings = []
        valid_metadatas = []

        for i, emb in enumerate(chunk_embeddings):
            meta = chunk_metadatas[i]
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)

            # Basic validation (shape and type)
            if not isinstance(emb, np.ndarray) or emb.ndim != 1:
                 if emb.ndim == 2 and emb.shape[0] == 1: # Allow (1, dim) shape
                     emb = emb.reshape(-1) # Convert to 1D
                 else:
                    logger.error(f"Skipping chunk {i} due to invalid embedding shape: {emb.shape if hasattr(emb, 'shape') else type(emb)}")
                    continue

            if emb.shape[0] != self.index.d:
                logger.error(f"Skipping chunk {i} due to dimension mismatch: expected {self.index.d}, got {emb.shape[0]}")
                continue

            # Add the embedding (reshaped for FAISS) and its metadata
            processed_embeddings.append(emb.reshape(1, -1))
            valid_metadatas.append(meta)

        if not processed_embeddings:
             logger.warning("No valid embeddings found to add after processing.")
             return

        # Concatenate embeddings for batch addition to FAISS
        embeddings_np = np.vstack(processed_embeddings).astype('float32') # Ensure float32

        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        logger.info(f"Added {embeddings_np.shape[0]} embeddings to FAISS index. Index size now: {self.index.ntotal}")

        # Store corresponding metadata
        self.metadata.extend(valid_metadatas)
        logger.info(f"Added {len(valid_metadatas)} metadata entries. Total metadata entries: {len(self.metadata)}")

        # Sanity check (optional but recommended)
        if self.index.ntotal != len(self.metadata):
            logger.critical(
                f"CRITICAL: FAISS index size ({self.index.ntotal}) mismatch with metadata list size ({len(self.metadata)})!"
            )
            # Depending on severity, you might want to raise an error here


    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the vector store for the k most similar chunks.

        Args:
            query_vector: The numpy array representing the query embedding.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains:
            - The metadata dictionary of the relevant chunk.
            - The distance (e.g., L2 distance) score.
            Returns an empty list if the index is empty or no results are found.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on an empty vector store.")
            return [] # Return empty list if store is empty

        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        if query_vector.ndim == 1:
            # Ensure query is float32 and has shape (1, dimension) for FAISS search
            query_vector = query_vector.reshape(1, -1).astype('float32')
        elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
             query_vector = query_vector.astype('float32') # Ensure correct type
        else:
            logger.error(f"Invalid query vector shape: {query_vector.shape}")
            raise ValueError("Query vector must be 1D or have shape (1, dimension)")

        if query_vector.shape[1] != self.index.d:
             logger.error(f"Query vector dimension mismatch: expected {self.index.d}, got {query_vector.shape[1]}")
             raise ValueError(f"Query vector dimension mismatch: expected {self.index.d}, got {query_vector.shape[1]}")


        # Ensure k is not greater than the number of items in the index
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            return []

        logger.info(f"Performing FAISS search for k={actual_k} nearest neighbors.")
        distances, indices = self.index.search(query_vector, actual_k)

        results = []
        # indices[0] contains the list of indices of the nearest neighbors for the first (and only) query vector
        # distances[0] contains the corresponding distances
        for i, dist in zip(indices[0], distances[0]):
            if i != -1: # FAISS uses -1 for invalid indices (shouldn't happen with IndexFlatL2 unless k > ntotal was passed, which we handle)
                try:
                    # Retrieve the metadata using the index returned by FAISS
                    metadata = self.metadata[i]
                    results.append((metadata, float(dist))) # Store metadata dict and distance
                except IndexError:
                    logger.error(f"FAISS returned index {i}, which is out of bounds for metadata list (size {len(self.metadata)}). This indicates a serious inconsistency.")
                    # Handle this error appropriately - maybe continue, maybe raise
                    continue
            else:
                logger.warning("FAISS search returned -1 index.")


        logger.info(f"Search completed. Found {len(results)} results.")
        return results # Returns list of (metadata_dict, distance)


# Instantiate the vector store (ensure this is done only once, e.g., at application startup)
# Make sure the dimension matches your embedding model (e.g., 384 for all-MiniLM-L6-v2)
vector_store = VectorStore(dimension=384)