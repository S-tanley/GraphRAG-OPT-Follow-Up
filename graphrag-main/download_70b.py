import os
os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]:
    print(f"==== downloading {repo} ====", flush=True)
    p = snapshot_download(repo, allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model"])
    print(f"DONE {repo} -> {p}", flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
