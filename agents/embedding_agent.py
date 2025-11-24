from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from utils.config import Config


class EmbeddingAgent:
    """Agent for generating embeddings and managing ChromaDB"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collections
        self.jd_collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_JD
        )
        self.cv_collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_CV
        )
    
    def index_job_description(self, job_id: int, chunks: List[str], metadata: Dict):
        """Index job description chunks in ChromaDB"""
        embeddings = self.embeddings.embed_documents(chunks)
        
        ids = [f"jd_{job_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_id": i} for i in range(len(chunks))]
        
        self.jd_collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def index_cv(self, candidate_id: int, chunks: List[str], metadata: Dict):
        """Index CV chunks in ChromaDB"""
        embeddings = self.embeddings.embed_documents(chunks)
        
        ids = [f"cv_{candidate_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_id": i} for i in range(len(chunks))]
        
        self.cv_collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def search_similar_jd_chunks(self, query: str, job_id: int, n_results: int = 5) -> List[Dict]:
        """Search for similar JD chunks"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.jd_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"job_id": job_id}
        )
        
        return self._format_results(results)
    
    def search_similar_cv_chunks(self, query: str, candidate_ids: List[int] = None, 
                                n_results: int = 5) -> List[Dict]:
        """Search for similar CV chunks"""
        query_embedding = self.embeddings.embed_query(query)
        
        where_filter = {}
        if candidate_ids:
            where_filter = {"candidate_id": {"$in": candidate_ids}}
        
        results = self.cv_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return self._format_results(results)
    
    def get_cv_context(self, candidate_id: int) -> str:
        """Get all chunks for a candidate as context"""
        results = self.cv_collection.get(
            where={"candidate_id": candidate_id}
        )
        
        if results and results['documents']:
            return "\n\n".join(results['documents'])
        return ""
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results"""
        formatted = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        return formatted
    
    def clear_job_data(self, job_id: int):
        """Clear all data for a job"""
        try:
            self.jd_collection.delete(where={"job_id": job_id})
        except:
            pass
    
    def clear_candidate_data(self, candidate_id: int):
        """Clear all data for a candidate"""
        try:
            self.cv_collection.delete(where={"candidate_id": candidate_id})
        except:
            pass


# Global instance
embedding_agent = EmbeddingAgent()