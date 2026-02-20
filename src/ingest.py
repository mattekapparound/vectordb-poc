from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from settings import settings

DOCS = [
    {"id": 1, "text": "Qdrant è un database vettoriale open-source, ottimo per similarity search."},
    {"id": 2, "text": "Docker Compose permette di avviare rapidamente servizi locali per sviluppo."},
    {"id": 3, "text": "Gli embedding trasformano testo in vettori numerici per confronti semantici."},
    {"id": 4, "text": "Python è un linguaggio popolare per prototipare pipeline di RAG."},
]

def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    texts = [d["text"] for d in DOCS]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    points = [
        PointStruct(
            id=d["id"],
            vector=v,
            payload={"text": d["text"]},
        )
        for d, v in zip(DOCS, vectors)
    ]

    client.upsert(
        collection_name=settings.collection_name,
        points=points,
    )

    print(f"✅ Upserted {len(points)} points into {settings.collection_name}")

if __name__ == "__main__":
    main()