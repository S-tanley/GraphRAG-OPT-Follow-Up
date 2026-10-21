#!/bin/bash
cd /Users/bowenzheng/Projects/GraphRAG-OPT-Follow-Up
.venv-mac/bin/python experiments/kv_prefetch/rag_mac.py gen --model mlx-community/Qwen3-8B-4bit --n 100
.venv-mac/bin/python experiments/kv_prefetch/rag_mac.py gen --model mlx-community/Qwen3-30B-A3B-4bit --n 100
.venv-mac/bin/python experiments/kv_prefetch/rag_mac.py retrieve --n 200
echo K0MAC DONE
