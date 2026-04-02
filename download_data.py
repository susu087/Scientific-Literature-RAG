"""
下载 LitSearch 数据集
"""

from datasets import load_dataset
import json
import os

def download_data():
    """下载 LitSearch 数据集"""

    print("=" * 50)
    print("Downloading LitSearch dataset...")
    print("=" * 50)

    # 创建数据目录
    os.makedirs("data", exist_ok=True)

    # 1. 下载查询集
    print("\n1. Downloading queries...")
    query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
    print(f"   Downloaded {len(query_data)} queries")

    # 保存查询到本地
    with open("data/queries.json", "w", encoding="utf-8") as f:
        json.dump([q for q in query_data], f, indent=2, ensure_ascii=False)
    print("   Saved to data/queries.json")

    # 2. 下载清理后的语料库
    print("\n2. Downloading clean corpus...")
    corpus_clean = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
    print(f"   Downloaded {len(corpus_clean)} papers")

    # 保存样本查看结构
    with open("data/corpus_sample.json", "w", encoding="utf-8") as f:
        sample_size = min(3, len(corpus_clean))
        json.dump([corpus_clean[i] for i in range(sample_size)], f, indent=2, ensure_ascii=False)
    print("   Sample saved to data/corpus_sample.json")

    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)

    # 打印数据统计
    print("\nDataset Statistics:")
    print(f"  Queries: {len(query_data)}")
    print(f"  Corpus (clean): {len(corpus_clean)}")

    # 显示一条查询示例
    print("\nSample Query:")
    if len(query_data) > 0:
        sample = query_data[0]
        print(f"  Query: {sample.get('query', 'N/A')}")
        print(f"  Relevant papers: {sample.get('relevant_corpusids', [])}")

if __name__ == "__main__":
    download_data()