# AAAfinal：科学文献检索与 RAG 原型

本项目围绕科学文献搜索构建检索实验和命令行检索增强生成（RAG）原型。它以 [Princeton NLP 的 LitSearch 项目](https://github.com/princeton-nlp/LitSearch)及其[公开数据集](https://huggingface.co/datasets/princeton-nlp/LitSearch)为基础，在原有检索基准代码之外，加入了本仓库的 E5 检索实验、本地重排、分区检索实验、指标分析和问答原型。

**当前定位：研究与演示代码。** 可运行的主线是「问题处理 → E5 检索 → 本地 CrossEncoder 重排 → 构造论文上下文 → 生成带文献编号的答案」。各阶段按预设顺序执行；目前没有根据中间证据自主选择工具或决定是否继续检索的 agentic 工作流，也没有网页界面。

## 功能与状态

| 模块 | 当前实现 |
| --- | --- |
| 检索基线 | 保留上游的 BM25、GTR、Instructor、E5、GRIT 索引与评测代码；不同模型需要各自的依赖和算力。 |
| 本仓库的检索实验 | E5 结果分析；基于 KMeans 分区的 E5 检索，以及按分区分数差选择搜索分区数的规则。 |
| 重排 | `BAAI/bge-reranker-base` 本地 CrossEncoder 重排及 top-20/50/100 深度对比；保留上游 GPT 重排代码。 |
| 命令行 RAG | 规则式查询关键词提取、E5 检索、重排、上下文构造、模板或 API 答案生成，并输出候选文献。 |
| 批量演示 | 对 `test_questions.json` 中的问题运行上述 RAG 主线，保存逐题结果和汇总报告。 |
| `final` 混合流程 | 代码包含 E5 + BM25 的加权 RRF 融合与进一步重排，但引用了仓库中尚不存在的 `eval.reranking.reason_to_rank_lite`；当前提交不能把它当作完整可运行的流程。 |

这里的“自适应分区”是按预设阈值改变搜索范围的规则，不代表通用代理决策。RAG 演示的默认 `mock` 答案是模板文本，不是大模型生成的研究结论。

## 数据与目录

LitSearch 数据集由原论文作者提供，包含 597 条文献搜索查询。项目默认通过 Hugging Face `datasets` 加载 `princeton-nlp/LitSearch` 的 `query` 和 `corpus_clean` 配置；完整论文语料与模型索引**没有**提交到本仓库。

- `data/queries.json`：仓库中的查询数据副本。
- `data/corpus_sample.json`：**仅 3 篇论文的样本**，用于查看格式；不能替代完整语料做正式评测。
- `test_questions.json`：16 个用于 RAG 演示的问题，与 597 条 LitSearch 检索查询不是同一套完整评测。
- `eval/retrieval/`：上游检索器，以及本仓库的 E5 分区脚本。
- `eval/reranking/`：上游 GPT 重排与本仓库的本地重排。
- `rag/`：查询处理、检索封装、上下文、答案生成、单题和批量演示。
- `litsearch_aligned_metrics.py`、`compute_metrics.py`、`analyze_*.py`：检索结果指标和实验分析。

默认的 `retrieval_indices/`、`results/` 为本地生成目录，被 `.gitignore` 排除。运行前请从完整语料创建所需索引。若完整语料加载失败，RAG 代码可能退回到 3 篇样本；**已有完整索引与样本语料混用时，很多检索结果会缺少标题和摘要**，不应据此判断问答质量。

## 环境准备

建议在 Python 3.10 的独立环境中，从仓库根目录执行命令。构建完整 E5 索引的现有脚本把设备设为 CUDA，需要可用的 NVIDIA GPU；RAG 演示支持 `--device cpu`，但在 CPU 上编码完整语料和重排会很慢。首次使用会下载 LitSearch 数据和 Hugging Face 模型，需要相应网络、磁盘空间与内存。

```bash
git clone https://github.com/susu087/AAAfinal.git
cd AAAfinal
python -m venv venv
# Linux / macOS: source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install openai
```

`openai` 当前未列入 `requirements.txt`，但公共工具模块在导入时需要它，因此上面单独安装。仓库的 `requirements.txt` 固定了较早的依赖版本；`README_Codex.md` 记录了另一组研究环境版本，两者并非统一的锁定环境。若安装出现版本冲突，应按实际 Python、CUDA 和 PyTorch 环境调整并记录版本。GTR、Instructor、GRIT 等可选基线还需要相应模型及额外依赖；下面的主线只覆盖 E5 与本地重排。

可用 `python test_install.py` 检查部分依赖和数据集访问。该脚本并不验证整个 RAG 流程。

## 复现检索与本地重排

以下命令使用完整 `corpus_clean` 语料，从仓库根目录运行。多行命令采用 Bash 的 `\` 续行符；在 Windows PowerShell 中请把同一条命令写成一行。创建 E5 索引可能耗时较长，生成文件位于 `retrieval_indices/LitSearch.title_abstract.e5`。

```bash
python -m eval.retrieval.build_index --index_type e5 --key title_abstract
python -m eval.retrieval.evaluate_index --index_name LitSearch.title_abstract.e5
python -m eval.reranking.rerank_local \
  --retrieval_results_file results/retrieval/LitSearch.title_abstract.e5.jsonl \
  --max_k 50 --device cuda
python compute_metrics.py \
  --results_file results/reranking/LitSearch.title_abstract.e5.reranked_top50.jsonl
```

检索结果默认写入 `results/retrieval/`，重排结果写入 `results/reranking/`。`compute_metrics.py` 汇总与 LitSearch 分组方式对齐的 Recall@k；它**不是**生成答案的正确性评测。如需 BM25 索引，可另运行：

```bash
python -m eval.retrieval.build_index --index_type bm25 --key title_abstract
```

BM25 使用 NLTK 分词和停用词资源；缺失时脚本会尝试下载。`analyze_e5_rerank_depths.py` 需要事先分别生成 top-20、top-50 和 top-100 的重排文件。

## 运行单问题 RAG 演示

先完成 E5 索引构建，再运行默认的 `e5` 流程。下面的 `mock` 后端只检查检索、上下文与输出结构，不应作为真实问答效果展示：

```bash
python -m rag.run_rag_demo \
  --question "Which papers study knowledge distillation for language model compression?" \
  --retrieval_pipeline e5 --answer_backend mock --device cuda \
  --save_json results/rag_demo.json
```

若要调用 OpenAI 兼容的聊天接口生成答案，可设置与服务商对应的密钥、模型和接口地址。例如代码默认使用 AIHubMix 地址：

```bash
export AIHUBMIX_API_KEY="<your-key>"
python -m rag.run_rag_demo \
  --question "Which papers study knowledge distillation for language model compression?" \
  --retrieval_pipeline e5 --answer_backend api --device cuda \
  --answer_model gpt-4o-mini --answer_base_url https://aihubmix.com/v1
```

Windows PowerShell 可用 `$env:AIHUBMIX_API_KEY="<your-key>"` 设置环境变量。代码也读取 `OPENAI_API_KEY`；使用其他兼容服务时请相应修改 `--answer_base_url` 和 `--answer_model`，并注意服务费用及数据发送范围。`--answer_backend local` **尚未实现本地模型生成**，目前返回的是带提示语的 `mock` 答案。

输出中的 `[D1]` 等编号对应当前上下文里的候选论文。API 后端返回的 `references` 列出输入上下文的候选文献，代码尚未逐句核验模型答案中的引用是否真实支撑论断，因此应人工检查重要结论。

### 批量演示

```bash
python -m rag.run_four_case_eval --preset four --answer_backend mock --device cuda
```

默认读取 `test_questions.json`，`--preset four` 选择 Q1、Q6、Q11、Q16；不指定该参数时处理文件中的全部有效问题。默认输出到 `rag/results/all_case_eval/`。此脚本当前只接受 `mock` 或占位的 `local` 后端；汇总中的 `system_chain_is_stable` 仅依据是否成功运行及是否有候选引用，**不衡量答案事实正确性**。

### 可选：分区检索实验

以下实验使用已建好的 E5 索引；评测脚本同样采用现有的 CUDA E5 实现。

```bash
python -m eval.retrieval.build_e5_partitions --num_partitions 10
python -m eval.retrieval.evaluate_partitioned_e5 \
  --partition_file retrieval_partitions/LitSearch.title_abstract.e5.kmeans_k10.json \
  --adaptive
```

## 当前限制与后续方向

1. `final` 混合 RAG 路径缺少 `reason_to_rank_lite` 实现，不能作为当前仓库已验证的最终方案。
2. 默认答案为模板；本地生成未实现，API 生成依赖外部服务。
3. 目前没有检索结果充分性判断、自动改写与再次检索、工具选择循环或任务级代理评测。
4. 当前评测脚本主要评价检索指标或演示链路是否运行；若研究目标是问答质量，需要另建有标注的答案与引用评测集，并与固定 RAG 基线比较。

## 来源、许可与引用

本仓库基于 [Princeton NLP / LitSearch](https://github.com/princeton-nlp/LitSearch)。LitSearch 论文、597 条查询基准和上游检索代码属于原作者的工作；本 README 所述的本仓库实验与 RAG 原型是基于该项目的扩展，不应将原论文贡献归为本仓库作者。原仓库代码采用 [MIT License](https://github.com/susu087/AAAfinal/blob/main/LICENSE)，使用时应保留许可声明。

若使用 LitSearch 数据或基准，请引用原论文：

```bibtex
@inproceedings{ajith2024litsearch,
  title={LitSearch: A Retrieval Benchmark for Scientific Literature Search},
  author={Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}
```
