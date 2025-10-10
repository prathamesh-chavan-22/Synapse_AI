from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from contextlib import asynccontextmanager

from api.routes import router
from core.model_loader import model_loader_instance
from core.vector_db import vector_db_instance
from services.ingestion_service import ingestion_service_instance
from processing.image_analyzer import ImageAnalyzer
from processing.audio_transcriber import AudioTranscriber

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multimodal RAG System...")

    try:
        logger.info("Loading embedding model...")
        model_loader_instance.load_embedding_model()

        logger.info("Loading LLM model...")
        model_loader_instance.load_llm(use_4bit=True)

        logger.info("Loading vision model...")
        vision_model, vision_processor = model_loader_instance.load_vision_model()

        logger.info("Loading Whisper model...")
        whisper_pipeline = model_loader_instance.load_whisper_model()

        logger.info("Initializing vector database...")
        vector_db_instance.get_or_create_collection()

        image_analyzer = ImageAnalyzer(vision_model, vision_processor)
        audio_transcriber = AudioTranscriber(whisper_pipeline)
        ingestion_service_instance.set_models(image_analyzer, audio_transcriber)

        logger.info("All models loaded successfully. System is ready!")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

    yield

    logger.info("Shutting down Multimodal RAG System...")
    model_loader_instance.cleanup()


app = FastAPI(
    title="Multimodal RAG System",
    description="A Retrieval-Augmented Generation system supporting PDF, DOCX, images, and audio files",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(2)  # Wait for server to start
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
