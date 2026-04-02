"""Run a first-round representative validation on four selected questions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.run_rag_demo import format_rag_result, run_rag_case


DEFAULT_CASES = [
    {
        "case_id": "Q1",
        "question_type": "方法型",
        "question": "What methods are commonly used to compress large-scale language models using knowledge distillation techniques?",
    },
    {
        "case_id": "Q6",
        "question_type": "定义型",
        "question": "What is prompt tuning in pre-trained language models?",
    },
    {
        "case_id": "Q11",
        "question_type": "比较型",
        "question": "What is the difference between pruning, quantization, and knowledge distillation for model compression?",
    },
    {
        "case_id": "Q16",
        "question_type": "推荐型",
        "question": "Which context construction strategy is more suitable for citation-grounded academic question answering?",
    },
]


def _slugify_case(case_id: str) -> str:
    return case_id.lower().replace(" ", "_")


def _load_cases(test_question_file: str = "") -> List[Dict[str, str]]:
    if not test_question_file:
        return list(DEFAULT_CASES)

    with open(test_question_file, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    selected_ids = {case["case_id"] for case in DEFAULT_CASES}
    normalized_cases = []
    for item in loaded:
        case_id = str(item.get("case_id") or item.get("id") or "")
        question = str(item.get("question") or item.get("query") or "")
        if case_id in selected_ids and question:
            question_type = str(item.get("question_type") or item.get("type") or "")
            normalized_cases.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                }
            )

    return normalized_cases or list(DEFAULT_CASES)


def _build_case_summary(case: Dict[str, str], result: Dict[str, Any]) -> Dict[str, Any]:
    top_retrieved = result["retrieved_docs"][:5]
    top_reranked = result["reranked_docs"][:5]
    return {
        "case_id": case["case_id"],
        "question_type": case["question_type"],
        "question": result["question"],
        "query_understanding_result": result["query_understanding"],
        "top_retrieved_papers": top_retrieved,
        "top_reranked_papers": top_reranked,
        "final_answer": result["answer"],
        "references": result["references"],
    }


def _build_round_analysis(case_summaries: List[Dict[str, Any]], failures: List[Dict[str, str]]) -> Dict[str, Any]:
    success_count = len(case_summaries)
    stable = success_count == 4 and not failures and all(summary["references"] for summary in case_summaries)

    likely_good = []
    likely_risky = []
    for summary in case_summaries:
        question_type = summary["question_type"]
        has_refs = bool(summary["references"])
        if question_type in {"方法型", "定义型", "比较型"} and has_refs:
            likely_good.append(question_type)
        if question_type == "推荐型" or not has_refs:
            likely_risky.append(question_type)

    likely_good = sorted(set(likely_good))
    likely_risky = sorted(set(likely_risky))

    if stable:
        expansion_judgement = "可以继续扩展到 16 条，但建议先人工抽查这 4 条的引用相关性。"
    else:
        expansion_judgement = "暂不建议直接扩展到 16 条，应先处理失败样例或引用相关性问题。"

    return {
        "system_chain_is_stable": stable,
        "successful_cases": success_count,
        "failed_cases": failures,
        "better_performing_question_types": likely_good,
        "potentially_problematic_question_types": likely_risky,
        "expansion_recommendation": expansion_judgement,
    }


def _write_text_report(
    output_path: str,
    case_summaries: List[Dict[str, Any]],
    full_results: List[Dict[str, Any]],
    analysis: Dict[str, Any],
) -> None:
    lines = [
        "First-Round Four-Case Validation",
        "",
        "Pipeline:",
        "keywords query understanding + E5 retrieval + rerank(top50) + structured context + mock answer",
        "",
    ]

    for case, result in zip(case_summaries, full_results):
        lines.append(f'{case["case_id"]} {case["question_type"]}')
        lines.append(format_rag_result(result, show_context_chars=1800))
        lines.append("")

    lines.append("Round Analysis:")
    lines.append(json.dumps(analysis, ensure_ascii=False, indent=2))

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-round validation on four representative questions.")
    parser.add_argument("--test_question_file", type=str, default="test_questions.json")
    parser.add_argument("--output_dir", type=str, default=os.path.join("rag", "results", "four_case_eval"))
    parser.add_argument("--query_mode", type=str, default="keywords", choices=["original", "keywords"])
    parser.add_argument("--context_strategy", type=str, default="structured", choices=["plain", "structured"])
    parser.add_argument("--answer_backend", type=str, default="mock", choices=["mock", "local"])
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--rerank_top_k", type=int, default=50)
    parser.add_argument("--context_top_n", type=int, default=5)
    parser.add_argument("--per_doc_max_chars", type=int, default=1200)
    parser.add_argument("--retrieval_index_path", type=str, default=os.path.join("retrieval_indices", "LitSearch.title_abstract.e5"))
    parser.add_argument("--dataset_path", type=str, default="princeton-nlp/LitSearch")
    parser.add_argument("--local_corpus_path", type=str, default=os.path.join("data", "corpus_sample.json"))
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker_batch_size", type=int, default=4)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cases = _load_cases(args.test_question_file)

    full_results = []
    case_summaries = []
    failures = []

    for case in cases:
        try:
            result = run_rag_case(
                question=case["question"],
                query_mode=args.query_mode,
                context_strategy=args.context_strategy,
                answer_backend=args.answer_backend,
                top_k=args.top_k,
                rerank_top_k=args.rerank_top_k,
                context_top_n=args.context_top_n,
                per_doc_max_chars=args.per_doc_max_chars,
                retrieval_index_path=args.retrieval_index_path,
                dataset_path=args.dataset_path,
                local_corpus_path=args.local_corpus_path,
                reranker_model=args.reranker_model,
                reranker_batch_size=args.reranker_batch_size,
                reranker_max_length=args.reranker_max_length,
                device=args.device,
            )
            full_result = {
                "case_id": case["case_id"],
                "question_type": case["question_type"],
                **result,
            }
            full_results.append(full_result)
            case_summaries.append(_build_case_summary(case, full_result))

            case_output_path = os.path.join(args.output_dir, f'{_slugify_case(case["case_id"])}.json')
            with open(case_output_path, "w", encoding="utf-8") as file:
                json.dump(full_result, file, indent=2, ensure_ascii=False)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "error": str(exc),
                }
            )

    analysis = _build_round_analysis(case_summaries, failures)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_json = {
        "run_timestamp": timestamp,
        "pipeline": {
            "query_mode": args.query_mode,
            "retrieval": "E5",
            "rerank_top_k": args.rerank_top_k,
            "context_strategy": args.context_strategy,
            "answer_backend": args.answer_backend,
        },
        "case_summaries": case_summaries,
        "round_analysis": analysis,
    }
    if failures:
        summary_json["failures"] = failures

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2, ensure_ascii=False)

    report_txt_path = os.path.join(args.output_dir, "report.txt")
    _write_text_report(report_txt_path, case_summaries, full_results, analysis)

    print(f"Saved summary JSON to: {os.path.abspath(summary_json_path)}")
    print(f"Saved text report to: {os.path.abspath(report_txt_path)}")
    if failures:
        print("Failures:")
        print(json.dumps(failures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
