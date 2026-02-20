from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from settings import settings

def main() -> None:
    query = "Come funziona la ricerca semantica con i vettori?"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    results = client.search(
        collection_name=settings.collection_name,
        query_vector=qvec,
        limit=3,
        with_payload=True,
    )

    print(f"🔎 Query: {query}\n")
    for r in results:
        print(f"- score={r.score:.4f} id={r.id} text={r.payload.get('text')}")

if __name__ == "__main__":
    main()