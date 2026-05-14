import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM  # FIX 1 & 2


# =========================
# CONFIG
# =========================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "school_rules"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cập nhật path khớp với tên đã tải trong download_qwen.py
LLM_MODEL_PATH = "/content/naive_rag/models/qwen3-4b"

TOP_K = 3


# =========================
# LOAD MODELS
# =========================

def load_embedding_model():
    print("Đang load embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def load_qwen_model():
    print("Đang load Qwen model...")

    # FIX 1: AutoTokenizer thay vì AutoProcessor (text-only model)
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    # FIX 2: AutoModelForCausalLM thay vì AutoModelForImageTextToText
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    model.eval()

    print("Đã load xong Qwen.")
    return tokenizer, model


# =========================
# RETRIEVAL
# =========================

def load_chroma_collection():
    # FIX 4: khởi tạo ChromaDB một lần duy nhất, không tạo lại mỗi query
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def search_chroma(question: str, embedding_model, chroma_collection, top_k: int = 3):
    question_vector = embedding_model.encode(question).tolist()

    results = chroma_collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )

    return results


def build_context(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, doc in enumerate(documents):
        metadata = metadatas[i]
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        section = metadata.get("section", "unknown")

        context_parts.append(
            f"[Nguồn: {source} | Mục: {section} | Chunk: {chunk_index}]\n{doc}"
        )

    return "\n\n".join(context_parts)


# =========================
# GENERATION
# =========================

def remove_thinking_text(text: str) -> str:
    """Cắt bỏ phần <think>...</think> nếu model sinh ra."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "<think>" in text:
        text = text.split("<think>")[-1].strip()
    return text.strip()


def generate_answer(question: str, context: str, tokenizer, model):
    system_prompt = """Bạn là trợ lý hỏi đáp nội quy trường học.

Nhiệm vụ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không tự bịa thêm thông tin ngoài CONTEXT.
- Nếu CONTEXT không có thông tin phù hợp, hãy trả lời: "Tôi không tìm thấy thông tin này trong nội quy."
- Trả lời bằng tiếng Việt.
- Trả lời ngắn gọn, rõ ràng."""

    user_prompt = f"""CONTEXT:
{context}

CÂU HỎI:
{question}

Hãy trả lời câu hỏi dựa trên CONTEXT."""

    # FIX 3: content là string thuần, không phải list of dicts kiểu vision model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False  # FIX: tắt thinking mode của Qwen3
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,  # FIX: bỏ warning generation flags
            top_p=None,
            top_k=None
        )

    # Chỉ decode phần model sinh ra, bỏ phần prompt đầu vào
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    # FIX 1: tokenizer.decode thay vì processor.decode
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = remove_thinking_text(answer)

    return answer


# =========================
# MAIN
# =========================

def main():
    embedding_model = load_embedding_model()
    tokenizer, llm_model = load_qwen_model()

    # FIX 4: init ChromaDB một lần, truyền vào hàm search
    chroma_collection = load_chroma_collection()

    print("\nHệ thống RAG đã sẵn sàng.")
    print("Gõ 'exit', 'quit' hoặc 'q' để thoát.\n")

    while True:
        question = input("Nhập câu hỏi: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Đã thoát chương trình.")
            break

        if not question:
            continue

        # 1. Embed câu hỏi → search ChromaDB
        results = search_chroma(
            question=question,
            embedding_model=embedding_model,
            chroma_collection=chroma_collection,
            top_k=TOP_K
        )

        # 2. Build context từ kết quả RAG
        context = build_context(results)

        # 3. Qwen sinh câu trả lời
        answer = generate_answer(
            question=question,
            context=context,
            tokenizer=tokenizer,
            model=llm_model
        )

        print("\nTrả lời:", answer)
        print()


if __name__ == "__main__":
    main()