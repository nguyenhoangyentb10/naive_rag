import json
import chromadb


def store_documents_to_chroma(json_path: str):
    # Load file đã có vector
    with open(json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Tạo ChromaDB lưu local
    client = chromadb.PersistentClient(path="./chroma_db")

    # Tạo collection
    collection = client.get_or_create_collection(
        name="school_rules"
    )

    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for doc in documents:
        ids.append(doc["id"])
        texts.append(doc["text"])
        metadatas.append(doc["metadata"])
        embeddings.append(doc["vector"])

    # Add vào vector database
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Đã lưu {len(documents)} chunks vào ChromaDB")


if __name__ == "__main__":
    store_documents_to_chroma("school_rules_with_vectors.json")