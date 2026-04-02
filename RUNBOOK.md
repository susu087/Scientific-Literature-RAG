# Setup
source /root/autodl-tmp/susu/venv/bin/activate
export HF_HOME=/root/autodl-tmp/susu/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# Important Notes
- 项目根目录是 /root/autodl-tmp/susu
- 不要使用 /autodl-tmp/susu
- 不要默认进入其他子目录

# Retrieval / Rerank
- E5 检索脚本：eval/retrieval/e5.py
- 实际使用的重排脚本：eval/reranking/rerank_local.py
- 必须模块方式运行：
  python -m eval.reranking.rerank_local ...

# Main Experiment Conclusion
- E5 overall outperforms BM25
- rerank(top50) is the main recommended configuration
- top100 has stronger Recall@10 / Hit@10 but weaker MRR than top50

# Next Development Target
新增 RAG 原型模块，建议放在 rag/ 目录：
- rag/run_rag_demo.py
- rag/query_understanding.py
- rag/build_context.py
- rag/generate_answer.py
- rag/prompts.py
