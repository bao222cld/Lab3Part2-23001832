
# src/representations/word_embedder.py
import re
import numpy as np
import gensim.downloader as api

class WordEmbedder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        print(f"Loading model {model_name} ...")
        self.model = api.load(model_name)
        self.dim = self.model.vector_size

    def simple_tokenize(self, text: str):
        if not isinstance(text, str):
            return []
        text = text.lower()
        tokens = re.findall(r"[a-z0-9']+", text)
        return tokens

    def get_vector(self, word: str):
        if word is None:
            return None
        w = word.lower()
        if hasattr(self.model, "key_to_index"):
            in_vocab = w in self.model.key_to_index
        else:
            in_vocab = w in self.model.index_to_key
        if in_vocab:
            return self.model[w]
        return None

    def get_similarity(self, word1: str, word2: str):
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def get_most_similar(self, word: str, top_n: int = 10):
        w = (word or "").lower()
        try:
            return self.model.most_similar(w, topn=top_n)
        except Exception:
            return []

    def embed_document(self, document: str):
        tokens = self.simple_tokenize(document)
        vecs = [self.get_vector(t) for t in tokens if self.get_vector(t) is not None]
        if len(vecs) == 0:
            return np.zeros(self.dim, dtype=float)
        return np.mean(np.vstack(vecs), axis=0)
