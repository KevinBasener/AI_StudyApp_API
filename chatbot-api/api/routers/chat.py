import logging
import time
import traceback

import chromadb
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import (
    ChatHistory,
    ChatHistoryList,
    ChatHistoryResponse,
    ChatInput,
    ChatResponse,
    Document,
)
from services.ollama import generate_response
from services.vector_store import vector_store
from utils.document_utility import log_memory_usage, generate_embedding_with_timeout

router = APIRouter()
logger = logging.getLogger(__name__)

from services.vector_db_service import get_chroma_collection


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput, db: Session = Depends(get_db), collection: chromadb.Collection = Depends(get_chroma_collection)
):
    start_time = time.time()
    log_memory_usage()
    # Log the received document IDs
    logger.info(
        f"Chat endpoint called with message: '{chat_input.message}', model: {chat_input.model}, "
        f"document_ids: {chat_input.document_ids}"
    )

    # --- Validate Input document_ids (Optional but Recommended) ---
    if not chat_input.document_ids:
        # Although the model requires it, an empty list might be sent.
        raise HTTPException(status_code=400, detail="Document IDs list cannot be empty.")

    # --- 1. Generate Embedding for User Query ---
    try:
        # ... (embedding generation logic remains the same) ...
        logger.info("Generating embedding for the user query.")
        query_embedding_np = await generate_embedding_with_timeout(chat_input.message)
        if isinstance(query_embedding_np, np.ndarray):
            query_embedding = query_embedding_np.flatten().tolist()
        else:
            query_embedding = query_embedding_np
        logger.info("Query embedding generated.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # ... (error handling) ...
        raise HTTPException(status_code=500, detail="Failed to process query embedding.")

    # --- 2. Search ChromaDB Collection (with Filtering) ---
    try:
        # --- Build the ChromaDB 'where' filter ---
        # Convert integer IDs to strings to match metadata storage format
        doc_ids_as_strings = [str(doc_id) for doc_id in chat_input.document_ids]
        where_filter = {
            "doc_id": {
                "$in": doc_ids_as_strings
            }
        }
        logger.info(f"Querying ChromaDB collection '{collection.name}' with filter: {where_filter}")

        # --- Perform the filtered query ---
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=15,  # Retrieve up to 15 relevant chunks matching the filter
            where=where_filter,  # Apply the filter here
            include=["metadatas", "documents", "distances"]
        )
        num_results = len(search_results["ids"][0]) if search_results and search_results.get("ids") else 0
        logger.info(f"ChromaDB query returned {num_results} potential chunks matching the filter.")

    except Exception as e:
        # ... (error handling for ChromaDB query) ...
        raise HTTPException(status_code=500, detail="Error searching relevant document chunks.")

    # --- 3. Assemble Context from Filtered Chunks ---
    # The rest of the context assembly logic (Steps 3, 4, 5, 6, 7)
    # remains exactly the same as in the previous version, as it operates
    # on the `search_results` which now contain only the filtered chunks.
    context = ""
    total_chars = 0
    MAX_CONTEXT_CHARS = 20000
    added_chunks_info = []

    if not search_results or not search_results.get("ids") or not search_results["ids"][0]:
        context = "No relevant document sections found matching the specified document IDs."  # Adjusted message
        logger.warning("No relevant chunks found during filtered vector search.")
    else:
        context_parts = []
        logger.info("Assembling context from filtered ChromaDB search results...")
        # Access the results for the first query (index 0)
        retrieved_metadatas = search_results["metadatas"][0]
        retrieved_documents = search_results["documents"][0]
        retrieved_distances = search_results["distances"][0]

        for i in range(len(retrieved_metadatas)):
            metadata = retrieved_metadatas[i]
            chunk_text = retrieved_documents[i]
            distance = retrieved_distances[i]

            if not chunk_text or chunk_text.isspace():
                logger.warning(f"Skipping result with empty document/text content: {metadata}")
                continue

            chunk_filename = metadata.get("filename", "Unknown Source")
            chunk_index = metadata.get("chunk_index", '?')
            original_doc_id = metadata.get("doc_id", 'Unknown')  # Already filtered, but good for header

            chunk_header = f"Source: {chunk_filename} (DocID: {original_doc_id}, Chunk: {chunk_index}, Distance: {distance:.4f})"

            estimated_len = len(chunk_header) + len(chunk_text) + 100

            if total_chars + estimated_len <= MAX_CONTEXT_CHARS:
                context_parts.append(f"{chunk_header}\n\n{chunk_text}")
                total_chars += estimated_len
                added_chunks_info.append(f"{chunk_filename} Chk {chunk_index} (Dist: {distance:.4f})")
            else:
                logger.warning(
                    f"Reached context character limit ({total_chars}/{MAX_CONTEXT_CHARS}). "
                    f"Stopping chunk inclusion. Included {len(context_parts)} chunks."
                )
                break

        if not context_parts:
            context = "Found relevant document sections matching the specified IDs, but couldn't fit any within the context window size."  # Adjusted message
            logger.warning("No chunks could be fitted into the context window.")
        else:
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Assembled context with {len(context_parts)} chunks. Total estimated chars: {total_chars}")
            logger.info(f"Included chunks from: {', '.join(added_chunks_info)}")

    # --- 4. Prepare Prompt for LLM ---
    # ... (Same prompt structure) ...
    prompt = (
        "You are a helpful AI assistant..."
        f"Context:\n{context}\n\n"
        f"User Question: {chat_input.message}\n\n"
        "Assistant Answer:"
    )

    # --- 5. Generate Response using Ollama ---
    # ... (Same LLM call logic) ...
    try:
        response_text = await generate_response(chat_input.model, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate response from AI model.")

    # --- 6. Save Chat History ---
    # ... (Same history saving logic) ...
    try:
        chat_history = ChatHistory(user_input=chat_input.message, bot_response=response_text, model=chat_input.model)
        db.add(chat_history)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save chat history: {e}")


    # --- 7. Return Response ---
    # ... (Same return) ...
    finally:
        log_memory_usage()
        logger.info(f"Total chat processing took {time.time() - start_time:.2f} seconds")

    return ChatResponse(message=response_text)


@router.get("/chat/history", response_model=ChatHistoryList)
def get_chat_history(db: Session = Depends(get_db)) -> ChatHistoryList:
    logger.debug("get_chat_history called")
    history = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
    logger.debug(f"Retrieved {len(history)} chat history items")
    return ChatHistoryList(
        chat_history=[ChatHistoryResponse.model_validate(item) for item in history]
    )
