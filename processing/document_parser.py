import PyPDF2
import docx
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Parsing PDF file: {file_path}")
            chunks = []

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    if text.strip():
                        chunks.append({
                            "text": text,
                            "page_number": page_num + 1,
                            "total_pages": total_pages
                        })

            logger.info(f"Extracted {len(chunks)} pages from PDF")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise

    @staticmethod
    def parse_docx(file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Parsing DOCX file: {file_path}")
            chunks = []

            doc = docx.Document(file_path)
            full_text = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)

            combined_text = "\n".join(full_text)

            if combined_text.strip():
                chunks.append({
                    "text": combined_text,
                    "paragraph_count": len(full_text)
                })

            logger.info(f"Extracted {len(full_text)} paragraphs from DOCX")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing DOCX: {str(e)}")
            raise

    @staticmethod
    def parse_txt(file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Parsing TXT file: {file_path}")
            chunks = []

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            if text.strip():
                chunks.append({
                    "text": text
                })

            logger.info(f"Extracted text from TXT file")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing TXT: {str(e)}")
            raise

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        try:
            if not text or not text.strip():
                return []

            text = text.strip()
            chunks = []

            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + chunk_size

                if end < text_length:
                    last_period = text.rfind('.', start, end)
                    last_newline = text.rfind('\n', start, end)
                    last_space = text.rfind(' ', start, end)

                    split_point = max(last_period, last_newline, last_space)

                    if split_point > start:
                        end = split_point + 1

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                start = end - chunk_overlap

            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    @staticmethod
    def get_file_type(file_path: str) -> str:
        path = Path(file_path)
        extension = path.suffix.lower()

        file_type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'txt',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.m4a': 'audio',
            '.flac': 'audio',
            '.ogg': 'audio'
        }

        return file_type_mapping.get(extension, 'unknown')
