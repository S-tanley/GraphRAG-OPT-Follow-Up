"""
GraphRAG with Path-First KV Caching — main entry point.

Usage examples
--------------
# Fastest meaningful accuracy check (NeurIPS-standard metrics, beam search)
python main.py --fast_mode neurips --retrieve --path_enum

# Match the original RoG paper exactly (256 shuffled samples, beam=4)
python main.py --fast_mode original --retrieve --path_enum

# With Subgraph KV Cache (paper Section 5)
python main.py --fast_mode neurips --retrieve --path_enum --use_pfkv

# Batch-optimized retrieval + PFKV
python main.py --fast_mode neurips --retrieve --path_enum --batch_opt --use_pfkv

# Data-parallel: one model replica per GPU, prompts distributed across them
python main.py --fast_mode neurips --retrieve --parallel_mode data_parallel --num_gpus 4

# Model-parallel: model layers spread across GPUs (use when model > 1 GPU)
python main.py --fast_mode neurips --retrieve --parallel_mode model_parallel --device_map balanced
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from config import QUICK_MODE_CONFIGS, set_reproducible_seed
from generation import PathFirstGraphRAGGeneration, format_prompt
from graph_utils import path_to_str
from metrics import evaluate_metrics
from metaqa_adapter import load_metaqa
from retrieval import BatchOptimizedRetrieval, SimpleGraphRAGRetrieval


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GraphRAG with Path-First KV Caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset / model
    parser.add_argument("--dataset", default="rmanluo/RoG-cwq",
                        help=(
                            "HuggingFace dataset id OR one of the MetaQA shortcuts: "
                            "metaqa-1hop | metaqa-2hop | metaqa-3hop  "
                            "(default: rmanluo/RoG-cwq)"
                        ))
    parser.add_argument("--metaqa_dir", default="metaQA",
                        help="Path to local MetaQA directory containing kb.txt and N-hop/ "
                             "subdirs (only used when --dataset is metaqa-*)")
    parser.add_argument("--metaqa_split", default="test",
                        choices=["train", "dev", "test"],
                        help="MetaQA split to load (default: test)")
    parser.add_argument("--llm_model", default="rmanluo/RoG",
                        help="HuggingFace model for generation (default: rmanluo/RoG)")
    parser.add_argument("--use_auth_token", default=None,
                        help="HuggingFace auth token for gated models")

    # Batch / size
    parser.add_argument("--batch_num", type=int, default=64,
                        help="Samples per batch (overridden by --fast_mode)")
    parser.add_argument("--test_size", type=int, default=None,
                        help="Total samples to run (overridden by --fast_mode)")
    parser.add_argument("--seed", type=int, default=42)

    # Fast-mode presets
    parser.add_argument(
        "--fast_mode",
        choices=list(QUICK_MODE_CONFIGS.keys()),
        default=None,
        help="Preset: " + " | ".join(
            f"{k} ({v['description']})" for k, v in QUICK_MODE_CONFIGS.items()
        ),
    )

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens (default: 50 for neurips/original, 256 otherwise)")
    parser.add_argument("--num_beams", type=int, default=None,
                        help="Beam width (default: 4 for neurips/original, 1 (greedy) otherwise)")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Disable standard KV caching")

    # Retrieval
    parser.add_argument("--retrieve", action="store_true",
                        help="Enable graph-path retrieval stage")
    parser.add_argument("--hop_constraint", type=int, default=3,
                        help="Max hops for path finding")
    parser.add_argument("--path_enum", action="store_true",
                        help="Use Algorithm-1 optimized path enumeration instead of BFS")
    parser.add_argument("--batch_opt", action="store_true",
                        help="Enable batch deduplication (requires --path_enum)")
    parser.add_argument("--max_paths_per_query", type=int, default=None,
                        help="Cap paths per query (for fair comparison)")

    # PFKV cache (Section 5)
    parser.add_argument("--use_pfkv", action="store_true",
                        help="Enable Subgraph KV Cache with positional re-encoding")
    parser.add_argument("--recompute_policy", default="none",
                        choices=["none", "epic1", "epic2", "epic4",
                                 "cacheblend15", "cacheblend30", "full"],
                        help="Selective recompute for faithful PFKV reuse (Llama-family). "
                             "none=GraphRAG-Opt as written; full=GraphRAG-R (no reuse); "
                             "epic*/cacheblend*=GraphRAG-Opt + recompute")
    parser.add_argument("--pfkv_cache_size", type=int, default=1000)
    parser.add_argument("--run_sanity_checks", action="store_true",
                        help="Verify KV cache implementation on startup")

    # Model loading
    parser.add_argument("--device_map", default="auto",
                        help="HuggingFace device_map (default: auto)")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--no_use_fast_tokenizer", dest="use_fast_tokenizer",
                        action="store_false")

    # Multi-GPU
    parser.add_argument(
        "--parallel_mode", default="none",
        choices=["none", "data_parallel", "model_parallel"],
        help=(
            "none            – single GPU (default)\n"
            "data_parallel   – N independent replicas, prompts round-robin'd across GPUs\n"
            "model_parallel  – layers spread across GPUs via device_map"
        ),
    )
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs for data_parallel / model_parallel")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Explicit GPU IDs, e.g. '0,1,2'  (overrides --num_gpus)")

    # Output
    parser.add_argument("--store", action="store_true",
                        help="Save per-sample results to JSON")

    # Sharding (for fast multi-GPU full-scale eval: one single-GPU worker per shard)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out_preds", default=None,
                        help="Write {questions, answers, predictions} JSON for aggregation")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Parallelism reconciliation
    if args.parallel_mode == "data_parallel" and args.device_map:
        print("[parallel] data_parallel overrides --device_map → None")
        args.device_map = None
    if args.gpu_ids and args.num_gpus:
        print("[parallel] --gpu_ids set; ignoring --num_gpus")

    # GPU info
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[GPU] {n} GPU(s): {names}")
        print(f"[GPU] Parallelism: {args.parallel_mode}")
    else:
        print("[GPU] No CUDA — running on CPU")

    # Reproducibility
    set_reproducible_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Resolve size / generation params from fast_mode preset
    if args.fast_mode:
        cfg = QUICK_MODE_CONFIGS[args.fast_mode]
        test_size  = cfg["samples"]
        batch_size = cfg["batch_size"]
        if args.max_new_tokens is None:
            args.max_new_tokens = cfg.get("max_new_tokens", 256)
        if args.num_beams is None:
            args.num_beams = cfg.get("num_beams", 1)
        print(f"Fast mode '{args.fast_mode}': {cfg['description']}")
        print(f"  {test_size} samples, batch {batch_size}, "
              f"beams={args.num_beams}, max_tokens={args.max_new_tokens}")
    else:
        test_size  = args.test_size or 512
        batch_size = args.batch_num
        if args.max_new_tokens is None:
            args.max_new_tokens = 256
        if args.num_beams is None:
            args.num_beams = 1
        print(f"Standard mode: {test_size} samples, batch {batch_size}")

    # Dataset
    print(f"Loading dataset: {args.dataset}")
    _metaqa_shortcuts = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if args.dataset in _metaqa_shortcuts:
        hops = _metaqa_shortcuts[args.dataset]
        # If local files are present, use them (works for all 3 hops).
        # Otherwise fall back to the HuggingFace mirror (3-hop only).
        from pathlib import Path as _Path
        local_kb = _Path(args.metaqa_dir) / "kb.txt"
        if local_kb.exists():
            dataset = load_metaqa(
                hops=hops,
                split=args.metaqa_split,
                data_dir=args.metaqa_dir,
            )
        elif hops == 3:
            print("[MetaQA] Local files not found — loading from HuggingFace (mhivanov123/metaqa-3hop)")
            hf_split = "validation" if args.metaqa_split == "dev" else args.metaqa_split
            dataset = load_dataset("mhivanov123/metaqa-3hop", split=hf_split)
        else:
            raise FileNotFoundError(
                f"MetaQA local files not found at {args.metaqa_dir}. "
                f"For {args.dataset}, local files are required. "
                "Download from https://github.com/yuyuz/MetaQA"
            )
        if len(dataset) > test_size:
            dataset = dataset.shuffle(seed=args.seed).select(range(test_size))
    else:
        dataset = load_dataset(args.dataset, split="test")
        if args.fast_mode == "original":
            dataset = dataset.shuffle(seed=42).select(range(256))
        else:
            dataset = dataset.shuffle(seed=args.seed).select(range(min(test_size, len(dataset))))

    # Shard the dataset across workers (each worker = one GPU, single-GPU batched).
    if args.num_shards > 1:
        idx = [i for i in range(len(dataset)) if i % args.num_shards == args.shard_id]
        dataset = dataset.select(idx)
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(dataset)} samples")

    # Generator
    use_kv_cache = not args.no_kv_cache
    generator = PathFirstGraphRAGGeneration(
        args.llm_model,
        use_kv_cache=use_kv_cache,
        use_pfkv=args.use_pfkv,
        pfkv_cache_size=args.pfkv_cache_size,
        use_auth_token=args.use_auth_token,
        device_map=args.device_map,
        run_sanity_checks=args.run_sanity_checks,
        use_fast_tokenizer=args.use_fast_tokenizer,
        parallel_mode=args.parallel_mode,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        recompute_policy=args.recompute_policy,
    )

    # Algorithm label
    algo = "Baseline BFS"
    if args.path_enum:
        algo = "Algorithm-1 Path Enumeration"
        if args.batch_opt:
            algo += " + Batch Dedup"
    if args.use_pfkv:
        algo += " + PFKV Cache"
    print(f"Algorithm: {algo}")
    if args.max_paths_per_query:
        print(f"Path cap: {args.max_paths_per_query} per query")

    if args.store:
        Path(f"results/{args.dataset}").mkdir(parents=True, exist_ok=True)

    # ── Batch loop ───────────────────────────────────────────────────────────
    total_samples = len(dataset)
    num_batches   = (total_samples + batch_size - 1) // batch_size

    retrieve_times, generation_times, e2e_times, throughputs = [], [], [], []
    total_hits = total_f1 = total_acc = 0.0
    total_processed = 0
    retrieval_stats: dict = {}
    all_questions, all_answers, all_predictions = [], [], []

    print(f"Processing {total_samples} samples in {num_batches} batches")
    print("=" * 80)

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        e2e_start = time.time()
        start_idx = batch_idx * batch_size
        end_idx   = min(start_idx + batch_size, total_samples)

        batch_sample  = dataset[start_idx:end_idx]
        batch_samples = dataset.select(range(start_idx, end_idx))

        queries = batch_sample["question"]
        answers = batch_sample["answer"]
        prompts: list = []
        batch_paths: list = []

        # ── Retrieval ────────────────────────────────────────────────────
        if args.retrieve:
            retrieve_start = time.time()

            if args.path_enum:
                retriever = BatchOptimizedRetrieval(batch_samples, device="cuda")
                results, stats = retriever.retrieve_batch(
                    args.hop_constraint, args.batch_opt, args.max_paths_per_query)
            else:
                retriever = SimpleGraphRAGRetrieval(batch_samples)
                results, stats = retriever.retrieve_batch(
                    args.hop_constraint, args.max_paths_per_query)

            for k, v in stats.items():
                retrieval_stats[k] = retrieval_stats.get(k, 0) + v

            retrieve_time = time.time() - retrieve_start
            retrieve_times.append(retrieve_time)

            # Optional: save retrieved paths
            if args.store:
                suffix = ("optimized" if args.path_enum else "baseline")
                if args.batch_opt:    suffix += "_batch"
                if args.use_pfkv:     suffix += "_pfkv"
                if args.max_paths_per_query:
                    suffix += f"_max{args.max_paths_per_query}"
                path_file = f"results/{args.dataset}/{suffix}_paths.json"
                if batch_idx == 0:
                    open(path_file, "w").close()
                with open(path_file, "a") as f:
                    for r in results:
                        f.write(json.dumps({
                            "question": r["query"],
                            "paths": [path_to_str(p) for p in r["paths"]],
                        }) + "\n")

            for r in results:
                prompts.append(format_prompt(r["query"], r["paths"], args.llm_model))
                batch_paths.append(r["paths"])
        else:
            retrieve_time = 0.0
            for q in queries:
                prompts.append(format_prompt(q, [], args.llm_model))
                batch_paths.append([])

        # ── Generation ───────────────────────────────────────────────────
        gen_start = time.time()
        predictions = generator.generate_batch(
            prompts,
            batch_paths if (args.use_pfkv and args.retrieve) else None,
            queries=list(queries) if (args.use_pfkv and args.retrieve) else None,
        )
        generation_time = time.time() - gen_start
        generation_times.append(generation_time)

        prompt_tokens = sum(len(generator.tokenizer.encode(p)) for p in prompts)
        output_tokens = sum(len(generator.tokenizer.encode(p)) for p in predictions)
        throughput = (prompt_tokens + output_tokens) / generation_time if generation_time > 0 else 0
        throughputs.append(throughput)

        # ── Metrics ──────────────────────────────────────────────────────
        m = evaluate_metrics(predictions, answers)
        if args.out_preds:
            all_questions.extend(list(queries))
            all_answers.extend([list(a) if not isinstance(a, str) else a for a in answers])
            all_predictions.extend(list(predictions))
        n = len(predictions)
        total_hits += m["hits_at_1"] * n
        total_f1   += m["f1"]        * n
        total_acc  += m["accuracy"]  * n
        total_processed += n

        e2e_time = time.time() - e2e_start
        e2e_times.append(e2e_time)

        # Optional: save per-sample results
        if args.store:
            suffix = ("optimized" if (args.retrieve and args.path_enum) else
                      "baseline"  if args.retrieve else "direct")
            if args.batch_opt:   suffix += "_batch"
            if args.use_pfkv:    suffix += "_pfkv"
            if args.max_paths_per_query:
                suffix += f"_max{args.max_paths_per_query}"
            out_file = f"results/{args.dataset}/{suffix}_results.json"
            if batch_idx == 0:
                open(out_file, "w").close()
            with open(out_file, "a") as f:
                for i in range(len(queries)):
                    f.write(json.dumps({
                        "question": queries[i],
                        "answer":   answers[i],
                        "prediction": predictions[i],
                    }) + "\n")

        print(f"Batch {batch_idx+1}/{num_batches}  "
              f"Retrieve: {retrieve_time:.2f}s  "
              f"Generate: {generation_time:.2f}s  "
              f"Hits@1: {m['hits_at_1']:.4f}  "
              f"F1: {m['f1']:.4f}  "
              f"Acc(substr): {m['accuracy']:.4f}")

    # Dump predictions for cross-shard aggregation
    if args.out_preds:
        import json as _json
        with open(args.out_preds, "w") as _f:
            _json.dump({"questions": all_questions, "answers": all_answers,
                        "predictions": all_predictions}, _f)
        print(f"[shard {args.shard_id}/{args.num_shards}] wrote {len(all_predictions)} preds -> {args.out_preds}")

    # ── Final report ─────────────────────────────────────────────────────────
    n = total_processed or 1
    hits_final = total_hits / n
    f1_final   = total_f1   / n
    acc_final  = total_acc  / n

    print("\n" + "=" * 80)
    print(f"RESULTS — {algo}")
    print("=" * 80)
    print(f"Samples: {total_processed}  |  Seed: {args.seed}  |  "
          f"Dataset: {args.dataset}  |  Model: {args.llm_model}")
    print(f"beams={args.num_beams}  max_tokens={args.max_new_tokens}  "
          f"retrieval={'yes' if args.retrieve else 'no'}")
    print("-" * 80)
    print("PERFORMANCE:")
    if retrieve_times:
        print(f"  Avg retrieval   : {np.mean(retrieve_times):.2f}s")
    print(f"  Avg generation  : {np.mean(generation_times):.2f}s")
    print(f"  Avg end-to-end  : {np.mean(e2e_times):.2f}s")
    print(f"  Avg throughput  : {np.mean(throughputs):.1f} tok/s")
    if generator.ttfts:
        print(f"  Avg TTFT        : {np.mean(generator.ttfts):.4f}s")
        print(f"  Median TTFT     : {np.median(generator.ttfts):.4f}s")
    print("-" * 80)
    print("QUALITY (NeurIPS-standard):")
    print(f"  Hits@1  (primary)  : {hits_final:.4f}   ← report this in the paper")
    print(f"  F1     (token F1)  : {f1_final:.4f}")
    print(f"  Acc  (substr/old)  : {acc_final:.4f}  (lenient, for reference only)")

    # PFKV stats
    if args.use_pfkv and generator.pfkv_cache:
        s = generator.pfkv_cache.get_cache_statistics()
        print("\nPFKV CACHE:")
        print(f"  Size: {s['cache_size']}  Hits: {s['cache_hits']}  "
              f"Misses: {s['cache_misses']}  Hit-rate: {s.get('hit_rate', 0):.2%}")

    # Retrieval stats
    if args.retrieve and retrieval_stats:
        print("\nRETRIEVAL:")
        if args.path_enum:
            print(f"  Pruning:  {retrieval_stats.get('graph_pruning_time', 0):.2f}s  "
                  f"Paths found: {retrieval_stats.get('paths_found', 0)}")
            if args.batch_opt:
                print(f"  Pair redundancy: {retrieval_stats.get('pair_redundancy_ratio', 0):.1%}  "
                      f"Efficiency: {retrieval_stats.get('efficiency_gain', 1):.2f}x")
        else:
            print(f"  BFS paths: {retrieval_stats.get('total_paths', 0)}  "
                  f"Time: {retrieval_stats.get('processing_time', 0):.2f}s")

    print("=" * 80)
    print("Done — results are reproducible with the same seed.")

    if args.use_pfkv:
        if generator.using_subgraph_cache:
            print("\nSubgraph KV Cache (Section 5): position-agnostic storage + RoPE realignment ENABLED")
        else:
            print("\nLegacy PathFirstKVCache in use (subgraph_kv_cache.py not found)")


if __name__ == "__main__":
    main()
