from sentence_transformers import SentenceTransformer
import json

# Load file chunks đã tạo trước đó
with open("school_rules_chunks.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Embedding từng chunk
for doc in documents:
    text = doc["text"]
    vector = model.encode(text).tolist()
    doc["vector"] = vector

# Lưu lại file có vector
with open("school_rules_with_vectors.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("Đã embedding xong")