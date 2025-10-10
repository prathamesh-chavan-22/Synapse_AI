import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, persist_directory: str = "./data/chromadb"):
        logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")

        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection_name = "multimodal_rag_collection"
        self.collection = None

    def get_or_create_collection(self, embedding_function=None):
        try:
            if embedding_function:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name
                )

            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            return self.collection

        except Exception as e:
            logger.error(f"Error creating/getting collection: {str(e)}")
            raise

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call get_or_create_collection() first.")

        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(texts)} documents to collection")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def search(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call get_or_create_collection() first.")

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )

            logger.info(f"Search completed, found {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def delete_by_filename(self, filename: str) -> int:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call get_or_create_collection() first.")

        try:
            results = self.collection.get(
                where={"filename": filename}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f"Deleted {deleted_count} documents for filename: {filename}")
                return deleted_count
            else:
                logger.info(f"No documents found for filename: {filename}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call get_or_create_collection() first.")

        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Collection '{self.collection_name}' has been deleted")
            self.collection = None

        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise


vector_db_instance = VectorDatabase()
