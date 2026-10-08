"""Evidence retriever: TF-IDF + cosine similarity (default, zero native deps).

FAISS is used automatically when importable (IndexFlatIP over the same
TF-IDF vectors). A light topic-boost reranker refines the top-k. Every hit
carries its snippet id, title, topic and source so agents can cite it.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .kb.snippets import SNIPPETS, EVAL_QUERIES

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except Exception:
    _faiss = None
    _FAISS_AVAILABLE = False

FAISS_AVAILABLE = _FAISS_AVAILABLE


class EvidenceRetriever:
    def __init__(self, snippets=None):
        self.snippets = snippets or SNIPPETS
        self.texts = [f"{s['title']}. {s['text']}" for s in self.snippets]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.backend = "tfidf-cosine"
        self._index = None
        if _FAISS_AVAILABLE:
            try:
                dense = self.matrix.toarray().astype(np.float32)
                norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9
                dense = dense / norms
                self._index = _faiss.IndexFlatIP(dense.shape[1])
                self._index.add(dense)
                self.backend = "faiss"
            except Exception:
                self._index = None
                self.backend = "tfidf-cosine"

    def _scores(self, query):
        q = self.vectorizer.transform([query])
        if self._index is not None:
            dense = q.toarray().astype(np.float32)
            dense = dense / (np.linalg.norm(dense) + 1e-9)
            D, I = self._index.search(dense, len(self.snippets))
            scores = np.zeros(len(self.snippets))
            scores[I[0]] = D[0]
            return scores
        return cosine_similarity(q, self.matrix)[0]

    def retrieve(self, query, k=5, topic_boost=None):
        """Return top-k hits: [{id,title,text,topic,source,score}]."""
        scores = self._scores(query).astype(float)
        if topic_boost:
            for i, s in enumerate(self.snippets):
                if s["topic"] in topic_boost:
                    scores[i] *= 1.25
        order = np.argsort(scores)[::-1][:k]
        hits = []
        for i in order:
            s = self.snippets[int(i)]
            hits.append({
                "id": s["id"], "title": s["title"], "text": s["text"],
                "topic": s["topic"], "source": s["source"],
                "score": float(scores[int(i)]),
            })
        return hits

    def rerank(self, query, hits):
        """Light rerank: prefer hits sharing query keywords with the title."""
        qwords = set(query.lower().split())
        scored = []
        for h in hits:
            overlap = len(qwords & set(h["title"].lower().split()))
            scored.append((h["score"] + 0.05 * overlap, h))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [h for _, h in scored]

    # ---------------- evaluation ----------------
    def precision_at_k(self, k=3):
        precisions = []
        for item in EVAL_QUERIES:
            hits = self.retrieve(item["query"], k=k)
            got = {h["id"] for h in hits}
            want = set(item["relevant"])
            precisions.append(len(got & want) / max(1, k))
        return float(np.mean(precisions))

    def citation_correctness(self, k=3):
        """Fraction of eval queries where >=1 retrieved hit is relevant."""
        ok = 0
        for item in EVAL_QUERIES:
            hits = self.retrieve(item["query"], k=k)
            got = {h["id"] for h in hits}
            if got & set(item["relevant"]):
                ok += 1
        return ok / max(1, len(EVAL_QUERIES))
