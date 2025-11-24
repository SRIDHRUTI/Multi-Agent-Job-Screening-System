import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    """Configuration settings for the job screening system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///job_screening.db")
    
    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_JD = "job_descriptions"
    CHROMA_COLLECTION_CV = "candidate_cvs"
    
    # Embedding Model
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # LLM Model
    LLM_MODEL = "gpt-4o-mini"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Matching Thresholds
    MIN_MATCH_SCORE = 60.0  # Minimum score to shortlist
    TOP_N_CANDIDATES = 10
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Email Settings (for demo purposes)
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@example.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return True

# Validate config on import
Config.validate()
