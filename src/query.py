import chromadb
from sentence_transformers import SentenceTransformer


def search(question: str, top_k: int = 3):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    question_vector = model.encode(question).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_collection(
        name="school_rules"
    )

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )

    return results


if __name__ == "__main__":
    while True:
        question = input("\nNhập câu hỏi của bạn, gõ 'exit' để thoát: ")

        if question.lower() == "exit":
            print("Đã thoát chương trình.")
            break

        results = search(question, top_k=3)

        print("\nCác đoạn nội quy tìm được:")

        for i, text in enumerate(results["documents"][0], start=1):
            metadata = results["metadatas"][0][i - 1]

            print(f"\n--- Kết quả {i} ---")
            print("Text:", text)
            print("Metadata:", metadata)