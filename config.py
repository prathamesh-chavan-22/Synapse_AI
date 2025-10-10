import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    PROJECT_ROOT = Path(__file__).parent

    DATA_DIR = PROJECT_ROOT / "data"
    INGESTED_FILES_DIR = DATA_DIR / "ingested"
    CHROMADB_DIR = DATA_DIR / "chromadb"

    INGESTED_FILES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "Salesforce/blip-image-captioning-base")
    WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-base")

    USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))

    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "rag_system.log")

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    ALLOWED_FILE_EXTENSIONS = [
        '.pdf', '.docx', '.doc', '.txt',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',
        '.mp3', '.wav', '.m4a', '.flac', '.ogg'
    ]

    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

    SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("VITE_SUPABASE_SUPABASE_ANON_KEY")


config = Config()
