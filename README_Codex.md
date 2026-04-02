# Project Goal
毕业设计：面向科学文献的大模型检索增强方法设计与实现

# Current Status
已完成：
- BM25 retrieval
- E5 retrieval
- local reranking with bge-reranker-base
- rerank depth ablation: top20 / top50 / top100
- group based analysis

当前主方案：
- E5 + rerank(top50)

# Important Paths
- project root: /root/autodl-tmp/susu
- venv: /root/autodl-tmp/susu/venv/bin/activate

# Environment
source /root/autodl-tmp/susu/venv/bin/activate
export HF_HOME=/root/autodl-tmp/susu/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# Verified Versions
- python 3.10
- torch 2.1.2+cu121
- numpy 1.26.4
- sentence-transformers 2.7.0
- transformers 4.41.2

# Existing Key Scripts
- eval/retrieval/e5.py
- eval/reranking/rerank.py
- eval/reranking/rerank_local.py
- compute_metrics.py
- analyze_bm25_by_group.py
- analyze_e5_by_group.py
- analyze_e5_vs_rerank_by_group.py
- analyze_e5_rerank_depths.py
- utils/openai_utils.py
- utils/utils.py

# Notes
- 实际使用的是 local cross-encoder reranker
- rerank 脚本请用模块方式运行：
  python -m eval.reranking.rerank_local ...
- 不要改动原 rerank.py 的既有逻辑
- 当前论文主推方案是 E5 + rerank(top50)

# Next Goal
实现最小端到端 RAG 原型：
question -> query understanding -> E5 retrieval -> rerank(top50) -> context building -> answer generation -> citations
