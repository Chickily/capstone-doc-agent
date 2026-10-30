import uuid
import chromadb


class VectorStore:
    """One in-memory Chroma collection per user session (embeds with Chroma's default MiniLM)."""

    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=f"docs_{uuid.uuid4().hex[:12]}",   # unique so sessions don't share data
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        self.delete_source(chunks[0]["metadata"]["source"])  # re-upload replaces old version
        for i in range(0, len(chunks), 128):
            batch = chunks[i:i + 128]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

    def search(self, query: str, k: int = 5) -> list[dict]:
        n = self.collection.count()
        if n == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, n))
        return [
            {"text": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()
