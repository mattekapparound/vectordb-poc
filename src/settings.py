import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = os.getenv("COLLECTION_NAME", "demo_collection")
    vector_size: int = int(os.getenv("VECTOR_SIZE", "384"))

settings = Settings()