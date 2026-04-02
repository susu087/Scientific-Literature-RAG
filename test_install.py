# test_install.py
import sys
print(f"Python路径: {sys.executable}")
print(f"Python版本: {sys.version}")

# 测试导入
try:
    import datasets
    print(f"✓ datasets: {datasets.__version__}")
except Exception as e:
    print(f"✗ datasets: {e}")

try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import nltk
    print(f"✓ nltk: {nltk.__version__}")
except Exception as e:
    print(f"✗ nltk: {e}")

# 测试数据集加载
print("\n测试数据集加载...")
try:
    from datasets import load_dataset
    # 只加载一个小样本测试
    ds = load_dataset("princeton-nlp/LitSearch", "query", split="full[:1]")
    print(f"✓ 成功加载数据集，包含 {len(ds)} 个样本")
    print(f"  示例问题: {ds[0]['query'][:50]}...")
except Exception as e:
    print(f"✗ 加载失败: {e}")