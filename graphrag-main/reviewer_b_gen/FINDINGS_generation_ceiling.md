# Generation-speedup honest ceiling (definitive, 2026-06-14)

GLOBAL cross-batch path reuse (full datasets, unbounded cache, max_paths<=30):
  CWQ: 3531 q, 43115 path-instances, 25157 distinct -> 41.7% reuse -> 1.71x prefill reduction
  WEB: 1628 q, 16882 inst, 14383 distinct -> 14.8% -> 1.17x
  MQA: 14274 q, 197739 inst, 184796 distinct -> 6.5% -> 1.07x
Top path frequency: CWQ 59x, WEB 10x, MQA 5x (a few popular paths; long tail unique).

=> The honest ceiling for cross-query subgraph KV reuse is ~1.7x (CWQ), ~1.2x (WEB),
   ~1.07x (MQA) -- bounded by the DATA's path redundancy. No engine/kernel/framing can
   exceed it. The paper's 3.12x generation / 2.5x TTFT is NOT reachable from honest reuse
   on these datasets; the saved profiler artifacts (profile_results/) show the paper's
   "graphrag_kv vs graphrag" speedup is the STANDARD decode use_cache on/off toggle
   (PFKV=False asserted), not subgraph reuse. The B4 batch-throughput 2-5x I briefly
   reported was a token-double-counting artifact (broadcast counted the union prefix xB
   in the numerator while wall-clock was 0.5-0.9x); RETRACTED.
HONEST generation contribution: eliminates redundant prefill work W_red (1.4-1.7x,
   growing with batch/global reuse), accuracy-preserving (Opt~=R). Realized as larger
   wall-clock throughput only by the contiguous shared-KV-region kernel (cf. CacheBlend
   2.8-5x throughput, SubGCache 6.68x TTFT in prefill-dominated regimes).
