#!/bin/bash
echo "============================================"
echo "  RAG Web App v3 (FastAPI + Uvicorn)"
echo "  Embedding: BAAI/bge-m3 (568M, multilingual)"
echo "  LLM: Qwen/Qwen2.5-7B-Instruct (7B, fp16)"
echo "  VRAM cần: ~16GB (GPU L4 24GB ✓)"
echo "============================================"
echo ""

echo "===== [1/3] Cài đặt thư viện ====="
pip install -q fastapi "uvicorn[standard]" python-multipart \
    "sentence-transformers>=3.0.0" chromadb \
    transformers accelerate huggingface_hub PyMuPDF python-docx torch \
    --break-system-packages 2>/dev/null || \
pip install -q fastapi "uvicorn[standard]" python-multipart \
    "sentence-transformers>=3.0.0" chromadb \
    transformers accelerate huggingface_hub PyMuPDF python-docx torch

echo ""
echo "===== [2/3] Tải models trước (lần đầu ~15 phút) ====="
python3 -c "
from sentence_transformers import SentenceTransformer
print('⏳ Downloading BGE-M3 embedding model (~2GB)...')
m = SentenceTransformer('BAAI/bge-m3')
print(f'✅ BGE-M3 loaded (dim={m.get_sentence_embedding_dimension()})')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'⏳ Downloading LLM: {model_name} (~14GB)...')
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

used = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'✅ LLM loaded — VRAM: {used:.1f}GB / {total:.1f}GB')
"

echo ""
echo "===== [3/3] Khởi động server ====="
echo ""
echo "  🌐 Web UI:    http://localhost:7860"
echo "  📖 API Docs:  http://localhost:7860/docs"
echo ""
python3 -m uvicorn app:app --host 0.0.0.0 --port 7860 --log-level info