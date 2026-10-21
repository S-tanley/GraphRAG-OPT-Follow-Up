#!/bin/bash
# Clean, self-consistent re-run of the Figure 6 workload characterization.
# RoG = Llama-2-7B (6.7B). RAG/RAG+KV are hop-independent (run once per dataset);
# GraphRAG/GraphRAG+KV swept over hops 1..4. Output is RENAMED per (dataset,tag)
# after each invocation because workload.py's filename omits hop/config.
set -e
PY=/home/am137688/anaconda3/envs/graphrag/bin/python
cd /home/am137688/amir/graphRag/graphrag_kv
export CUDA_VISIBLE_DEVICES=0
OUT=workload_fig6
mkdir -p $OUT
S=32; B=8; TOK=128; SEED=42

run () {  # $1=dataset short, $2=dataset arg, $3=tag, $4...=configs
  local short="$1"; local ds="$2"; local tag="$3"; local hop="$4"; shift 4
  $PY workload.py --dataset "$ds" --configs "$@" \
      --samples $S --batch_size $B --max_tokens $TOK --seed $SEED \
      --hop $hop --out_dir $OUT --device 0
  mv "$OUT/${short}_RoG_N${S}_B${B}_b1_workload.json" "$OUT/${short}_${tag}.json"
  mv "$OUT/${short}_RoG_N${S}_B${B}_b1_workload.csv"  "$OUT/${short}_${tag}.csv"
}

# RoG-CWQ
run RoG-cwq    "rmanluo/RoG-cwq" rag       3 rag rag_kv
for H in 1 2 3 4; do run RoG-cwq "rmanluo/RoG-cwq" "gr_h${H}" $H graphrag graphrag_kv; done
# MetaQA-3hop
run metaqa-3hop "metaqa-3hop"    rag       3 rag rag_kv
for H in 1 2 3 4; do run metaqa-3hop "metaqa-3hop" "gr_h${H}" $H graphrag graphrag_kv; done

echo "ALL DONE -> $OUT"
