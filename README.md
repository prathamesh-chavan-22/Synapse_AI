# Synapse AI

Synapse AI is an advanced multimodal Retrieval-Augmented Generation (RAG) system with a React frontend, enabling intelligent querying across documents, images, and audio files.

## Features

- **Multimodal Document Processing**: Ingest and index PDFs, Word documents, text files, images, and audio
- **Advanced Text Extraction**: Smart chunking with semantic coherence
- **Image Captioning**: Automatic caption generation using Vision models
- **Audio Transcription**: Speech-to-text using Whisper
- **Vector Search**: Efficient similarity search using ChromaDB
- **LLM-Powered Answers**: Context-grounded responses with source citations
- **4-bit Quantization**: Efficient model loading for resource-constrained environments
- **RESTful API**: FastAPI-based endpoints for ingestion and querying
- **Web Interface**: Modern React-based frontend for easy file upload and querying

## System Architecture

```
├── api/                    # FastAPI routes and endpoints
├── core/                   # Core components (model loader, vector DB)
├── processing/            # File processing modules
├── services/              # Business logic (ingestion, query)
├── static/                # React frontend static files
├── data/                  # Data storage
│   ├── ingested/         # Uploaded files
│   └── chromadb/         # Vector database
├── main.py               # Application entry point
├── schemas.py            # Pydantic models
├── config.py             # Configuration settings
└── requirements.txt      # Python dependencies
```

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB+ RAM (32GB recommended for larger models)
- 10GB+ disk space for models and data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables (optional):
Create a `.env` file:
```
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VISION_MODEL_NAME=Salesforce/blip-image-captioning-base
WHISPER_MODEL_NAME=openai/whisper-base
USE_4BIT_QUANTIZATION=true
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000` and automatically open your default web browser to the React frontend interface.

### API Endpoints

#### 1. Ingest a File
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "message": "File indexed successfully.",
  "chunks_created": 15
}
```

#### 2. Query the System
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the main findings?",
    "top_k": 5
  }'
```

Response:
```json
{
  "query_text": "What are the main findings?",
  "answer": "The main findings include... [1][2]",
  "sources": [
    {
      "citation_number": 1,
      "filename": "report.pdf",
      "content_type": "pdf_page",
      "details": {
        "page_number": 5,
        "text_snippet": "..."
      }
    }
  ]
}
```

#### 3. Health Check
```bash
curl http://localhost:8000/api/health
```

#### 4. Database Statistics
```bash
curl http://localhost:8000/api/stats
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Web Interface

The system includes a modern, responsive React-based web interface that provides an intuitive way to interact with the RAG system. Features include:

- File upload interface for documents, images, and audio files
- Real-time query interface with source citations
- Responsive design optimized for desktop and mobile
- Automatic browser launch upon server startup

Access the web interface at `http://localhost:8000` after starting the server.

## Supported File Types

- **Documents**: PDF, DOCX, DOC, TXT
- **Images**: PNG, JPG, JPEG, GIF, BMP
- **Audio**: MP3, WAV, M4A, FLAC, OGG

## Configuration

Key configuration parameters in `config.py`:

- `LLM_MODEL_NAME`: Language model for generation
- `EMBEDDING_MODEL_NAME`: Model for text embeddings
- `VISION_MODEL_NAME`: Model for image captioning
- `WHISPER_MODEL_NAME`: Model for audio transcription
- `CHUNK_SIZE`: Text chunk size (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `DEFAULT_TOP_K`: Default number of results (default: 5)
- `MAX_NEW_TOKENS`: Maximum tokens in LLM response (default: 512)
- `TEMPERATURE`: LLM temperature (default: 0.7)

## Model Loading

On startup, the system loads:
1. Embedding model (sentence-transformers)
2. LLM (Mistral-7B with 4-bit quantization)
3. Vision model (BLIP for image captioning)
4. Whisper model (for audio transcription)

Models are cached locally after first download.

## Performance Optimization

- **4-bit Quantization**: Reduces memory usage by ~4x
- **GPU Acceleration**: Automatic GPU detection and usage
- **Batch Processing**: Efficient embedding generation
- **ChromaDB**: Fast vector similarity search

## Error Handling

The system includes comprehensive error handling:
- Invalid file type detection
- File size validation
- Model loading failures
- Vector database errors
- LLM generation timeouts

## Logging

Logs are written to:
- Console (stdout)
- `rag_system.log` file

Configure log level in `config.py` or via `LOG_LEVEL` environment variable.

## Development

### Project Structure

- `core/model_loader.py`: ML model management
- `core/vector_db.py`: ChromaDB abstraction
- `processing/document_parser.py`: Document text extraction
- `processing/image_analyzer.py`: Image caption generation
- `processing/audio_transcriber.py`: Audio transcription
- `services/ingestion_service.py`: File processing pipeline
- `services/query_service.py`: RAG query pipeline
- `api/routes.py`: FastAPI endpoints

### Adding New File Types

1. Add extension to `ALLOWED_FILE_EXTENSIONS` in `config.py`
2. Implement parser in `processing/` module
3. Add processing method in `ingestion_service.py`
4. Update `get_file_type()` in `document_parser.py`

## Troubleshooting

### Out of Memory Errors
- Enable 4-bit quantization
- Use smaller models
- Reduce batch size
- Use CPU instead of GPU

### Model Download Issues
- Check internet connection
- Verify Hugging Face access
- Manually download models to cache

### Slow Performance
- Ensure GPU is being used
- Check CUDA installation
- Reduce `MAX_NEW_TOKENS`
- Use smaller embedding model

## License

This project is part of SIH2523 Problem Statement implementation.

## Version

1.0.0
