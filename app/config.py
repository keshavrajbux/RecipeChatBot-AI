from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Recipe ChatBot"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    REDIS_URL: str = "redis://localhost:6379"
    
    # Model Settings
    MODEL_PATH: str
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # RAG Settings
    VECTOR_STORE_PATH: str = "./data/vector_store"
    HYBRID_SEARCH_ALPHA: float = 0.7
    DEFAULT_SEARCH_RESULTS: int = 5
    
    # LLM Settings
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings() 