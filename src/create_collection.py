from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from settings import settings

def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # ricrea la collection per esperimenti
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=Distance.COSINE,
        ),
    )

    print(f"✅ Created collection: {settings.collection_name} (size={settings.vector_size}, cosine)")

if __name__ == "__main__":
    main()