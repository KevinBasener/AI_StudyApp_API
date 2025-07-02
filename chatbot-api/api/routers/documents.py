import asyncio
import contextlib
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chromadb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

import psutil
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database import get_db
from models import Document, DocumentListResponse, DocumentResponse


# Create a context manager for the ThreadPoolExecutor
from services.vector_db_service import get_chroma_collection

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db),
    collection: chromadb.Collection = Depends(get_chroma_collection)
):
    start_time = time.time()
    log_memory_usage()
    logger.info(f"Starting upload for file: {file.filename}")
    try:
        # --- 1. Extract Text ---
        # ... (same as before) ...
        logger.info(f"Extracting text from {file.filename}")
        content = await extract_text_from_file(file)
        logger.info(f"Extracted {len(content)} characters from {file.filename}")
        if not content or content.isspace():
             logger.warning(f"File {file.filename} contains no usable text content.")
             raise HTTPException(status_code=400, detail="File contains no text content.")

        # --- 2. Create Document Record in SQL (to get ID) ---
        # ... (same as before, make sure Document model doesn't expect embedding) ...
        logger.info("Creating document record in database...")
        document = Document(
            filename=file.filename,
            content=content,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Created document record with ID: {document.id} for {file.filename}")


        # --- 3. Chunk the Text ---
        # ... (same as before) ...
        logger.info("Chunking document content...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, length_function=len, is_separator_regex=False,
        )
        chunks = text_splitter.split_text(content)
        logger.info(f"Split content into {len(chunks)} chunks.")
        if not chunks:
            # ... (same handling as before) ...
            return {"message": "Document processed, but no text chunks were generated for indexing.", "document_id": document.id}

        # --- 4. Prepare Data for ChromaDB ---
        logger.info(f"Generating embeddings and preparing data for {len(chunks)} chunks...")
        embeddings_list = []
        metadatas_list = []
        documents_list = [] # List to hold the actual chunk text for ChromaDB
        ids_list = []       # List to hold unique IDs for each chunk

        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                logger.warning(f"Skipping empty chunk {i+1}/{len(chunks)} for {file.filename}")
                continue
            try:
                embedding_np = await generate_embedding_with_timeout(chunk)
                # Ensure embedding is a flat list of floats for ChromaDB
                if isinstance(embedding_np, np.ndarray):
                    embedding = embedding_np.flatten().tolist()
                else: # Assuming it might already be a list
                    embedding = embedding

                # Create a unique ID for each chunk
                chunk_id = f"doc_{document.id}_chunk_{i}"

                metadata = {
                    "doc_id": str(document.id), # Store original document ID as string
                    "filename": file.filename,
                    "chunk_index": i
                    # You can add other metadata here if needed
                }

                embeddings_list.append(embedding)
                metadatas_list.append(metadata)
                documents_list.append(chunk) # Add the chunk text itself
                ids_list.append(chunk_id)

                if (i + 1) % 50 == 0:
                     logger.info(f"Processed {i+1}/{len(chunks)} chunks for {file.filename}")

            except HTTPException as http_exc:
                 logger.error(f"HTTPException generating embedding for chunk {i} of {file.filename}: {http_exc.detail}")
                 raise http_exc
            except Exception as e:
                logger.error(f"Error generating embedding or preparing data for chunk {i} of {file.filename}: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed process chunk {i+1}")

        # --- 5. Add Chunks to ChromaDB Collection ---
        if ids_list: # Check if there's anything to add
            logger.info(f"Adding {len(ids_list)} chunks to ChromaDB collection '{collection.name}'...")
            try:
                # Use the injected collection object
                collection.add(
                    embeddings=embeddings_list,
                    metadatas=metadatas_list,
                    documents=documents_list, # Store chunk text directly
                    ids=ids_list # Provide unique IDs for each chunk
                )
                logger.info(f"Successfully added {len(ids_list)} chunks to ChromaDB.")
            except Exception as e:
                 logger.error(f"Failed to add chunks to ChromaDB for doc ID {document.id}: {str(e)}")
                 logger.error(traceback.format_exc())
                 # Consider rolling back DB commit for the Document record?
                 db.rollback() # Rollback SQL Document creation if vector indexing fails
                 db.delete(document) # Explicitly delete if rollback doesn't cascade
                 db.commit()
                 logger.warning(f"Rolled back database entry for Document ID {document.id} due to ChromaDB error.")
                 raise HTTPException(status_code=500, detail="Failed to index document chunks in vector store.")
        else:
            logger.warning(f"No valid chunks were processed for {file.filename}, nothing added to ChromaDB.")

        # --- 6. Return Success ---
        # ... (same as before) ...
        logger.info(f"Successfully processed and indexed {file.filename}")
        return {"message": "Document uploaded and chunks indexed successfully", "document_id": document.id}

    # ... (keep existing except/finally blocks, potentially add db.rollback() in generic except) ...
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error during upload of {file.filename}: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            # Attempt rollback in case of unexpected error after initial commit
            db.rollback()
            logger.info("Database transaction rolled back due to unexpected error.")
        except Exception as rb_e:
            logger.error(f"Failed to rollback database transaction: {rb_e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )
    finally:
        log_memory_usage()
        logger.info(f"Upload process for {file.filename} took {time.time() - start_time:.2f} seconds")



@router.get("/{doc_id}")
async def get_document(doc_id: int, db: Session = Depends(get_db)):
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": document.id,
            "filename": document.filename,
            "content": document.content,
            "timestamp": document.timestamp,
        }
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving document")


@router.get("/search", response_model=DocumentListResponse)
async def search_documents(
    query: str, db: Session = Depends(get_db)
) -> DocumentListResponse:
    query_embedding = generate_embedding(query)

    results = vector_store.search(query_embedding, k=5)

    document_ids = [doc_id for doc_id, _ in results]
    documents = db.query(Document).filter(Document.id.in_(document_ids)).all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents]
    )
