import logging
import json
from typing import List, Dict, Any
from core.model_loader import model_loader_instance
from core.vector_db import vector_db_instance
from schemas import Source, QueryResponse

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self):
        logger.info("QueryService initialized")

    async def process_query(self, query_text: str, top_k: int = 5) -> QueryResponse:
        try:
            logger.info(f"Processing query: {query_text[:100]}... (top_k={top_k})")

            query_embedding = model_loader_instance.generate_embeddings([query_text])

            search_results = vector_db_instance.search(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            if not search_results['ids'][0]:
                logger.warning("No relevant documents found for the query")
                return QueryResponse(
                    query_text=query_text,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[]
                )

            context_parts = []
            sources = []

            for idx, (doc_id, document, metadata, distance) in enumerate(zip(
                search_results['ids'][0],
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            )):
                citation_num = idx + 1

                source_info_str = metadata.get('source_info', '{}')
                source_info = json.loads(source_info_str)
                modality = metadata.get('modality', 'text')
                filename = metadata.get('filename', 'unknown')

                content_type = self._determine_content_type(modality, source_info)

                context_part = f"[{citation_num}] Source: {filename}"

                if 'page_number' in source_info:
                    context_part += f", Page: {source_info['page_number']}"
                elif 'timestamp_start' in source_info:
                    context_part += f", Timestamp: {source_info['timestamp_start']:.1f}s"

                context_part += f"\nContent: {document}\n---"
                context_parts.append(context_part)

                source_details = {
                    "text_snippet": document[:200] + "..." if len(document) > 200 else document,
                    "similarity_score": float(1 - distance)
                }
                source_details.update(source_info)

                sources.append(Source(
                    citation_number=citation_num,
                    filename=filename,
                    content_type=content_type,
                    details=source_details
                ))

            context = "\n".join(context_parts)

            prompt = self._build_prompt(context, query_text)

            answer = model_loader_instance.generate_text(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7
            )

            answer = self._clean_answer(answer, prompt)

            logger.info(f"Query processed successfully with {len(sources)} sources")

            return QueryResponse(
                query_text=query_text,
                answer=answer,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _determine_content_type(self, modality: str, source_info: Dict) -> str:
        if modality == 'image':
            return 'image_caption'
        elif modality == 'audio':
            return 'audio_transcript'
        elif 'page_number' in source_info:
            return 'pdf_page'
        else:
            return 'document_text'

    def _build_prompt(self, context: str, query_text: str) -> str:
        prompt_template = """<s>[INST] You are an expert AI assistant. Your task is to answer the user's question based ONLY on the provided context. Do not use any external knowledge. For each piece of information you use, you must cite the source using the format [citation_number].

CONTEXT:
---
{context}
---

USER QUESTION:
{query}

Provide a clear, comprehensive answer using ONLY the information from the context above. Always cite your sources with [number] format. [/INST]

ASSISTANT ANSWER:
"""

        return prompt_template.format(context=context, query=query_text)

    def _clean_answer(self, answer: str, prompt: str) -> str:
        if "ASSISTANT ANSWER:" in answer:
            answer = answer.split("ASSISTANT ANSWER:")[-1].strip()

        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()

        answer = answer.strip()

        return answer


query_service_instance = QueryService()
