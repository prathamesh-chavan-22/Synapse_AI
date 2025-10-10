import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json

from processing.document_parser import DocumentParser
from processing.image_analyzer import ImageAnalyzer
from processing.audio_transcriber import AudioTranscriber
from core.model_loader import model_loader_instance
from core.vector_db import vector_db_instance

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.document_parser = DocumentParser()
        self.image_analyzer = None
        self.audio_transcriber = None
        self.ingested_files_dir = "./data/ingested"
        os.makedirs(self.ingested_files_dir, exist_ok=True)

    def set_models(self, image_analyzer: ImageAnalyzer, audio_transcriber: AudioTranscriber):
        self.image_analyzer = image_analyzer
        self.audio_transcriber = audio_transcriber
        logger.info("Models set for IngestionService")

    async def ingest_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting ingestion for file: {filename}")

            file_type = self.document_parser.get_file_type(file_path)

            if file_type == 'unknown':
                raise ValueError(f"Unsupported file type for: {filename}")

            saved_path = os.path.join(self.ingested_files_dir, filename)
            shutil.copy2(file_path, saved_path)

            chunks_data = []

            if file_type == 'pdf':
                chunks_data = await self._process_pdf(saved_path, filename)
            elif file_type == 'docx':
                chunks_data = await self._process_docx(saved_path, filename)
            elif file_type == 'txt':
                chunks_data = await self._process_txt(saved_path, filename)
            elif file_type == 'image':
                chunks_data = await self._process_image(saved_path, filename)
            elif file_type == 'audio':
                chunks_data = await self._process_audio(saved_path, filename)

            if not chunks_data:
                raise ValueError(f"No content extracted from file: {filename}")

            texts = [chunk['text'] for chunk in chunks_data]
            embeddings = model_loader_instance.generate_embeddings(texts)

            metadatas = []
            for i, chunk in enumerate(chunks_data):
                metadata = {
                    "filename": filename,
                    "file_path": saved_path,
                    "modality": chunk.get('modality', 'text'),
                    "text_chunk": chunk['text'],
                    "chunk_id": f"{filename}_chunk_{i}",
                    "source_info": json.dumps(chunk.get('source_info', {})),
                    "ingestion_timestamp": datetime.utcnow().isoformat()
                }
                metadatas.append(metadata)

            vector_db_instance.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Successfully ingested {len(chunks_data)} chunks from {filename}")

            return {
                "status": "success",
                "filename": filename,
                "message": "File indexed successfully.",
                "chunks_created": len(chunks_data),
                "file_type": file_type
            }

        except Exception as e:
            logger.error(f"Error ingesting file {filename}: {str(e)}")
            raise

    async def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        logger.info(f"Processing PDF: {filename}")

        pages = self.document_parser.parse_pdf(file_path)
        chunks_data = []

        for page in pages:
            text_chunks = self.document_parser.chunk_text(page['text'])

            for chunk in text_chunks:
                chunks_data.append({
                    'text': chunk,
                    'modality': 'text',
                    'source_info': {
                        'page_number': page['page_number'],
                        'total_pages': page['total_pages']
                    }
                })

        return chunks_data

    async def _process_docx(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        logger.info(f"Processing DOCX: {filename}")

        doc_content = self.document_parser.parse_docx(file_path)
        chunks_data = []

        for content in doc_content:
            text_chunks = self.document_parser.chunk_text(content['text'])

            for chunk in text_chunks:
                chunks_data.append({
                    'text': chunk,
                    'modality': 'text',
                    'source_info': {
                        'paragraph_count': content.get('paragraph_count', 0)
                    }
                })

        return chunks_data

    async def _process_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        logger.info(f"Processing TXT: {filename}")

        txt_content = self.document_parser.parse_txt(file_path)
        chunks_data = []

        for content in txt_content:
            text_chunks = self.document_parser.chunk_text(content['text'])

            for chunk in text_chunks:
                chunks_data.append({
                    'text': chunk,
                    'modality': 'text',
                    'source_info': {}
                })

        return chunks_data

    async def _process_image(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        if self.image_analyzer is None:
            raise ValueError("Image analyzer not initialized")

        logger.info(f"Processing Image: {filename}")

        analysis = self.image_analyzer.analyze_image(file_path)
        caption = analysis['caption']

        chunks_data = [{
            'text': f"Image: {filename}. Caption: {caption}",
            'modality': 'image',
            'source_info': {
                'caption': caption,
                'width': analysis['width'],
                'height': analysis['height'],
                'format': analysis['format']
            }
        }]

        return chunks_data

    async def _process_audio(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        if self.audio_transcriber is None:
            raise ValueError("Audio transcriber not initialized")

        logger.info(f"Processing Audio: {filename}")

        transcription = self.audio_transcriber.transcribe(file_path, return_timestamps=True)
        full_text = transcription['text']

        text_chunks = self.document_parser.chunk_text(full_text)
        chunks_data = []

        for chunk in text_chunks:
            chunks_data.append({
                'text': chunk,
                'modality': 'audio',
                'source_info': {
                    'transcript': chunk
                }
            })

        return chunks_data


ingestion_service_instance = IngestionService()
