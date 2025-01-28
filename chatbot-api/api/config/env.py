import os  # Import os module

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://user:password@localhost/chatbot_db"
)  # Read from environment variable
OLLAMA_API_BASE = os.getenv(
    "OLLAMA_API_BASE", "http://localhost:11434"
)  # Read from environment variable
FRONTEND_URL = os.getenv(
    "FRONTEND_URL", "http://localhost:3000"
)  # Read from environment variable
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY",
                         "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")  # Read from environment variable
