import sys, time
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    t = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start {repo}", flush=True)
    p = snapshot_download(repo, max_workers=16,
                          allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.py", "*.tiktoken"])
    print(f"[{time.strftime('%H:%M:%S')}] done {repo} -> {p} ({(time.time()-t)/60:.1f} min)", flush=True)
