import json
import logging
import os

from config.env import GROQ_API_BASE, GROQ_API_KEY

import httpx
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_available_models() -> list:
    url = f"{GROQ_API_BASE}/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    logger.info(f"Attempting to connect to Groq at: {url}")
    logger.info(f"Current GROQ_API_BASE: {GROQ_API_BASE}")
    logger.info(f"Environment GROQ_API_KEY: {'Present' if GROQ_API_KEY else 'Missing'}")

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please set GROQ_API_KEY in the environment variables.",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"Fetching models from {url}")
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])
            logger.info(f"Fetched models: {models}")
            return models
    except httpx.ConnectError as ex:
        logger.error(f"Failed to connect to Groq at {url}. Is it running?")
        raise HTTPException(
            status_code=503,
            detail=f"Groq service is not available at {url}. Please check if it's running and accessible.",
        ) from ex
    except httpx.HTTPStatusError as ex:
        logger.error(f"HTTP error occurred: {ex}")
        raise HTTPException(
            status_code=ex.response.status_code,
            detail=f"Error from Groq service: {ex.response.text}",
        ) from ex
    except Exception as ex:
        logger.error(f"Unexpected error occurred: {ex}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while connecting to Groq: {str(ex)}",
        ) from ex


async def generate_response(model: str, prompt: str, timeout: float = 60.0) -> str:
    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please set GROQ_API_KEY in the environment variables.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            # Parse the response
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise ValueError("No valid response received from Groq API")

    except httpx.ReadTimeout:
        return "I'm sorry, but the response is taking longer than expected. Please try again or try a shorter prompt."
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        return f"An error occurred while communicating with the Groq API: {str(e)}"
    except httpx.ConnectError:
        logger.error("Failed to connect to Groq. Is it running?")
        return "I'm sorry, but I'm unable to connect to the language model service at the moment. Please try again later."
    except Exception as ex:
        logger.error(f"Unexpected error occurred: {ex}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while connecting to Groq: {str(ex)}",
        ) from ex
