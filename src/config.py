import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv()

# Project root directory: .../radiotherapy-rag-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories:
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Models
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "radtherapy_documents")

# Retrieval and chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4