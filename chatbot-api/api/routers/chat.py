import logging
from http.client import HTTPException

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import (
    ChatHistory,
    ChatHistoryList,
    ChatHistoryResponse,
    ChatInput,
    ChatResponse,
    Document,
)
from api.services.ollama import generate_response
from api.services.vector_store import vector_store
from api.utils.text_processing import generate_embedding

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput, db: Session = Depends(get_db)):
    logger.info("Chat endpoint called with message: %s", chat_input.message)

    try:
        # Fetch only the specified documents
        selected_documents = db.query(Document).filter(Document.id.in_(chat_input.document_ids)).all()

        if not selected_documents:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [
                    f"Document: {doc.filename}\nContent: {doc.content[:4000]}..."
                    for doc in selected_documents
                ]
            )

    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    # Prepare the prompt with context
    prompt = f"Context:\n{context}\n\nUser: {chat_input.message}\nAssistant: Based on the context provided (if any), "

    # Generate response using Ollama
    try:
        response = await generate_response(chat_input.model, prompt)
    except Exception as e:
        logger.error(f"Ollama generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response from AI")

    # Save chat history
    try:
        logger.info("Adding chat history to the database...")
        chat_history = ChatHistory(
            user_input=chat_input.message,
            bot_response=response,
            model=chat_input.model,
        )
        db.add(chat_history)
        db.commit()
    except Exception as e:
        db.rollback()  # Rollback on failure
        logger.error(f"Database commit failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save chat history")

    return ChatResponse(message=response)



@router.get("/chat/history", response_model=ChatHistoryList)
def get_chat_history(db: Session = Depends(get_db)) -> ChatHistoryList:
    logger.debug("get_chat_history called")
    history = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
    logger.debug(f"Retrieved {len(history)} chat history items")
    return ChatHistoryList(
        chat_history=[ChatHistoryResponse.model_validate(item) for item in history]
    )
