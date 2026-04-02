import nltk
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Any
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType

class BM25(KVStore):
    def __init__(self, index_name: str):
        super().__init__(index_name, 'bm25')

        try:
            # 检查 punkt 是否已存在
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            print("NLTK data found, skipping download")
        except LookupError:
            # 如果不存在才下载（但我们已经手动安装了，所以理论上不会执行到这里）
            print("NLTK data not found, attempting download...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words('english'))
        self._stemmer = nltk.stem.PorterStemmer().stem
        self.index = None   # BM25 index

    def _encode_batch(self, texts: str, type: TextType, show_progress_bar: bool = True) -> List[str]:
        # lowercase, tokenize, remove stopwords, and stem
        tokens_list = []
        for text in tqdm(texts, disable=not show_progress_bar):
            tokens = self._tokenizer(text.lower())
            tokens = [token for token in tokens if token not in self._stop_words]
            tokens = [self._stemmer(token) for token in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def _query(self, encoded_query: List[str], n: int) -> List[int]:
        top_indices = np.argsort(self.index.get_scores(encoded_query))[::-1][:n].tolist()
        return top_indices

    def clear(self) -> None:
        super().clear()
        self.index = None
    
    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        super().create_index(key_value_pairs)
        self.index = BM25Okapi(self.encoded_keys)
    
    def load(self, dir_name: str) -> None:
        super().load(dir_name)
        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words('english'))
        self._stemmer = nltk.stem.PorterStemmer().stem
        return self