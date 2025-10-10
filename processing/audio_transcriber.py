import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AudioTranscriber:
    def __init__(self, whisper_pipeline=None):
        self.whisper_pipeline = whisper_pipeline
        logger.info("AudioTranscriber initialized")

    def set_pipeline(self, whisper_pipeline):
        self.whisper_pipeline = whisper_pipeline
        logger.info("Whisper pipeline set for AudioTranscriber")

    def transcribe(self, audio_path: str, return_timestamps: bool = True) -> Dict[str, Any]:
        if self.whisper_pipeline is None:
            raise ValueError("Whisper pipeline not initialized. Load model first.")

        try:
            logger.info(f"Transcribing audio file: {audio_path}")

            result = self.whisper_pipeline(
                audio_path,
                return_timestamps=return_timestamps,
                chunk_length_s=30
            )

            transcript_text = result.get("text", "")

            logger.info(f"Transcription completed, length: {len(transcript_text)} characters")

            return {
                "text": transcript_text,
                "chunks": result.get("chunks", []) if return_timestamps else []
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    def transcribe_with_timestamps(self, audio_path: str) -> list:
        try:
            result = self.transcribe(audio_path, return_timestamps=True)

            timestamped_chunks = []
            for chunk in result.get("chunks", []):
                timestamped_chunks.append({
                    "text": chunk.get("text", ""),
                    "timestamp_start": chunk.get("timestamp", [0, 0])[0],
                    "timestamp_end": chunk.get("timestamp", [0, 0])[1]
                })

            return timestamped_chunks

        except Exception as e:
            logger.error(f"Error transcribing audio with timestamps: {str(e)}")
            raise
