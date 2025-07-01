import asyncio
import contextlib
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from fastapi import HTTPException
from utils.text_processing import generate_embedding

@contextlib.contextmanager
def get_executor():
    with ThreadPoolExecutor() as executor:
        yield executor


logger = logging.getLogger(__name__)
def log_memory_usage() -> None:
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

async def generate_embedding_with_timeout(content, timeout=60) -> Any:
    try:
        logger.info(
            f"Starting generate_embedding_with_timeout, timeout set to {timeout} seconds"
        )
        loop = asyncio.get_event_loop()
        with get_executor() as executor:
            embedding = await asyncio.wait_for(
                loop.run_in_executor(executor, generate_embedding, content),
                timeout=timeout,
            )
        logger.info("generate_embedding_with_timeout completed successfully")
        return embedding
    except asyncio.TimeoutError:
        logger.error("Embedding generation timed out")
        raise HTTPException(
            status_code=500, detail="Embedding generation timed out"
        ) from None
    except Exception as exc:
        logger.error(f"Error in generate_embedding_with_timeout: {str(exc)}")
        raise HTTPException(
            status_code=500, detail="Error generating embedding"
        ) from exc