import json, importlib
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
cands = ["mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8",
         "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/DeepSeek-V2-Chat-0628", "zai-org/GLM-4.5-Air",
         "meta-llama/Llama-4-Scout-17B-16E-Instruct", "databricks/dbrx-instruct",
         "tencent/Hunyuan-A13B-Instruct", "baidu/ERNIE-4.5-300B-A47B-PT", "Qwen/Qwen3-30B-A3B"]
import transformers
for r in cands:
    try:
        info = api.model_info(r, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin")))
        gated = info.gated
        try:
            cfg = json.load(open(hf_hub_download(r, "config.json")))
            c = cfg.get("text_config", cfg)
            E = c.get("num_local_experts") or c.get("n_routed_experts") or c.get("num_experts") or c.get("moe_num_experts") or (c.get("ffn_config") or {}).get("moe_num_experts")
            k = c.get("num_experts_per_tok") or c.get("moe_topk") or c.get("moe_k") or (c.get("ffn_config") or {}).get("moe_top_k")
            sh = c.get("n_shared_experts") or c.get("num_shared_experts") or c.get("moe_num_shared_experts") or ("yes" if c.get("shared_expert_intermediate_size") else 0)
            L = c.get("num_hidden_layers") or c.get("n_layers")
            mt = cfg.get("model_type")
            try:
                importlib.import_module(f"transformers.models.{mt.replace('-', '_')}")
                sup = "native"
            except Exception:
                sup = "custom-code" if cfg.get("auto_map") else "NOT in 5.2"
            print(f"{r:45s} {size/1e9:7.0f}GB gated={str(gated):6s} type={mt:14s} L={L} E={E} top{k} shared={sh} tf={sup}")
        except Exception as e:
            print(f"{r:45s} {size/1e9:7.0f}GB gated={gated} config-err {type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        print(f"{r:45s} ERR {type(e).__name__}: {str(e)[:90]}")
