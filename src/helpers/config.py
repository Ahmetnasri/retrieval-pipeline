from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    DEVICE: str

    APP_HOST: str
    APP_PORT: int
    WORKERS: int

    UVICORN_HOST: str
    UVICORN_PORT: int
    UVICORN_WORKERS: int
    
    # Qdrant
    QDRANT_HOST: str
    DB_COLLECTION_NAME_PHASE1: str
    DB_COLLECTION_NAME_PHASE2: str
    DB_LIMIT_PHASE1: int
    DB_LIMIT_PHASE2: int
    DB_DIMENSIONALITY_PHASE1: int
    DB_DIMENSIONALITY_PHASE2: int
    DB_TYPE: str

    # Embedding model
    EMBEDDING_MODEL_PHASE1: str
    EMBEDDING_MODEL_PHASE2: str
    EMBEDDING_MODEL_PATH_PHASE1: str
    EMBEDDING_MODEL_PATH_PHASE2: str

    # Path settings
    SAVE_INPUT_QUERY: str
    INPUT_QUERY_PATH: str
    DATABASE_PATH: str
    DATABASE_PATH_LIGHTGLUE: str

    # disk settings
    EXTRACTOR_MODEL: str
    NUM_PATCH: int
    DISK_KEYPOINTS: int
    THRESHOLDS: list[float]

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()