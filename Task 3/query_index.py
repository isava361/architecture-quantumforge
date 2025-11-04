import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def load_chunks_text(index_dir: Path, metadatas: List[Dict[str, Any]]) -> List[str]:
    chunks_path = index_dir / "chunks.jsonl"
    if chunks_path.exists():
        arr = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                arr.append(obj["text"])
        return arr

    texts_cache = {}
    out = []
    for m in metadatas:
        p = m["path"]
        if p not in texts_cache:
            try:
                texts_cache[p] = Path(p).read_text(encoding="utf-8")
            except Exception:
                texts_cache[p] = ""
        full_text = texts_cache[p]
        pos = m.get("position", {})
        if pos.get("type") == "char":
            start = int(pos.get("start", 0))
            out.append(full_text[start:start+1500])  # эвристика
        else:
            words = full_text.split()
            start_w = int(pos.get("start", 0))
            snippet = " ".join(words[start_w:start_w+250])
            out.append(snippet)
    return out

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def tokenize_ru(text: str) -> List[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default="index")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--query", type=str, required=False)
    ap.add_argument("--k", type=int, default=50, help="сколько кандидатов достать из FAISS")
    ap.add_argument("--topn", type=int, default=5, help="сколько отдать в финале")

    ap.add_argument("--hybrid_bm25", action="store_true", help="включить BM25+эмбеддинги")
    ap.add_argument("--alpha", type=float, default=0.7, help="вес эмбеддингов в гибриде (0..1)")

    ap.add_argument("--rerank_cross_encoder", type=str, default="", help="имя cross-encoder модели (например, cross-encoder/ms-marco-Multilingual-MiniLM-L12-v2)")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata = load_metadata(index_dir / "metadata.jsonl")

    st_model = SentenceTransformer(args.model)

    chunk_texts = load_chunks_text(index_dir, metadata)

    if not args.query:
        raise SystemExit("--query обязателен")

    q_emb = st_model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, args.k)
    faiss_scores = D[0]
    faiss_indices = I[0]

    hybrid_scores = None
    if args.hybrid_bm25:
        corpus_tokens = [tokenize_ru(t) for t in chunk_texts]
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = tokenize_ru(args.query)
        bm25_all = np.array(bm25.get_scores(q_tokens), dtype="float32")   # по всем чанкам
        bm25_cand = bm25_all[faiss_indices]
        bm25_cand_n = minmax_scale(bm25_cand)
        faiss_n = minmax_scale(faiss_scores)
        alpha = max(0.0, min(1.0, args.alpha))
        hybrid = alpha * faiss_n + (1 - alpha) * bm25_cand_n
        order = np.argsort(-hybrid)
        faiss_indices = faiss_indices[order]
        faiss_scores = hybrid[order]
        hybrid_scores = hybrid

    final_indices = faiss_indices
    if args.rerank_cross_encoder:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.rerank_cross_encoder)
        pairs = [(args.query, chunk_texts[idx]) for idx in final_indices]
        ce_scores = ce.predict(pairs)  # чем больше, тем релевантнее
        ce_scores = np.array(ce_scores, dtype="float32")
        order = np.argsort(-ce_scores)
        final_indices = final_indices[order]
        faiss_scores = ce_scores[order]  # для вывода покажем скор CE

    N = min(args.topn, len(final_indices))
    results = []
    for rank, idx in enumerate(final_indices[:N], start=1):
        m = metadata[idx]
        snippet = f"[{m['title']}] {m['path']} (chunk #{m['chunk_index']})"
        r = {
            "rank": rank,
            "score": float(faiss_scores[rank-1]),
            "id": m["id"],
            "path": m["path"],
            "title": m["title"],
            "chunk_index": m["chunk_index"],
            "snippet": snippet
        }
        if hybrid_scores is not None:
            r["hybrid_score"] = float(hybrid_scores[rank-1])
        results.append(r)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
