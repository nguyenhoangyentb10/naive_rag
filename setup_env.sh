#!/bin/bash

echo "===== 1. Cài thư viện ====="
pip install -q transformers sentence-transformers==2.7.0 chromadb python-docx huggingface_hub accelerate

echo "===== 2. Kéo toàn bộ project từ GitHub ====="
git clone https://github.com/TEN_BAN/TEN_REPO.git /content/naive_rag

echo "===== 3. Tải lại model Qwen (bắt buộc vì quá lớn cho GitHub) ====="
python3 /content/naive_rag/src/download_qwen.py

echo ""
echo "===== XONG! Chạy lệnh sau để bắt đầu ====="
echo "python /content/naive_rag/src/query_LLM.py"