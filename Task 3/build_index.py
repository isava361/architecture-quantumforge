#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LC = True
except Exception:
    _HAS_LC = False



def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def collect_documents(data_dir: Path, exts=(".txt", ".md")) -> List[Path]:
    files: List[Path] = []
    for root, _, filenames in os.walk(data_dir):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(Path(root) / name)
    return sorted(files)



def word_chunk(text: str,
               chunk_size_words: int = 200,
               chunk_overlap_words: int = 40
               ) -> List[Tuple[str, int]]:

    words = text.split()
    chunks: List[Tuple[str, int]] = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size_words, len(words))
        piece_words = words[i:j]
        if not piece_words:
            break
        chunks.append((" ".join(piece_words), i))
        step = chunk_size_words - chunk_overlap_words if chunk_size_words > chunk_overlap_words else chunk_size_words
        i += max(step, 1)
    return chunks


def lc_chunk(text: str,
             chunk_size_chars: int = 1200,
             chunk_overlap_chars: int = 200
             ) -> List[Tuple[str, int]]:

    if not _HAS_LC:
        raise RuntimeError("RecursiveCharacterTextSplitter недоступен. Установите langchain-text-splitters или не используйте --use_lc_splitter.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parts: List[str] = splitter.split_text(text)

    chunks: List[Tuple[str, int]] = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            # fallback: если по какой-то причине не нашли — используем текущий курсор
            idx = cursor
        chunks.append((part, idx))
        cursor = idx + len(part)
    return chunks



def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms



def main():
    parser = argparse.ArgumentParser(description="Индексация БЗ в FAISS с эмбеддингами all-MiniLM-L6-v2")
    parser.add_argument("--data_dir", type=str, default="data", help="Каталог с .txt/.md файлами")
    parser.add_argument("--out_dir", type=str, default="index", help="Куда сохранять индекс и метаданные")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Имя модели эмбеддингов")

    parser.add_argument("--use_lc_splitter", action="store_true", help="Использовать RecursiveCharacterTextSplitter (по символам)")
    parser.add_argument("--chunk_size_words", type=int, default=200, help="Размер чанка по словам (если без LC)")
    parser.add_argument("--chunk_overlap_words", type=int, default=40, help="Перекрытие чанков по словам")
    parser.add_argument("--lc_chunk_chars", type=int, default=1200, help="Размер чанка по символам (LC)")
    parser.add_argument("--lc_chunk_overlap_chars", type=int, default=200, help="Перекрытие чанков по символам (LC)")

    parser.add_argument("--store_chunks", action="store_true", help="Сохранять тексты чанков в index/chunks.jsonl")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    files = collect_documents(data_dir)
    if not files:
        raise SystemExit(f"Нет документов в {data_dir.resolve()}")

    t_chunk = time.perf_counter()
    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for fp in files:
        text = read_text_file(fp)
        if not text.strip():
            continue

        if args.use_lc_splitter:
            parts = lc_chunk(text, chunk_size_chars=args.lc_chunk_chars, chunk_overlap_chars=args.lc_chunk_overlap_chars)
            for i, (part, start_char) in enumerate(parts):
                chunks.append(part)
                metadatas.append({
                    "id": f"{fp.as_posix()}::chunk_{i}",
                    "path": fp.as_posix(),
                    "title": fp.name,
                    "chunk_index": i,
                    "chunk_len_words": len(part.split()),
                    "position": {"type": "char", "start": int(start_char)},
                    "source": fp.as_posix(),
                })
        else:
            parts = word_chunk(text, chunk_size_words=args.chunk_size_words, chunk_overlap_words=args.chunk_overlap_words)
            for i, (part, start_word) in enumerate(parts):
                chunks.append(part)
                metadatas.append({
                    "id": f"{fp.as_posix()}::chunk_{i}",
                    "path": fp.as_posix(),
                    "title": fp.name,
                    "chunk_index": i,
                    "chunk_len_words": len(part.split()),
                    "position": {"type": "word", "start": int(start_word)},
                    "source": fp.as_posix(),
                })

    chunk_time = time.perf_counter() - t_chunk

    if not chunks:
        raise SystemExit("После чанкинга не осталось текста. Проверьте входные файлы и параметры.")

    t_model = time.perf_counter()
    model = SentenceTransformer(args.model)
    model_time = time.perf_counter() - t_model

    t_emb = time.perf_counter()
    emb = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # для cosine через IP
    )
    emb_time = time.perf_counter() - t_emb

    dim = int(emb.shape[1])

    t_faiss = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss_write_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_write_path))
    faiss_time = time.perf_counter() - t_faiss

    t_write = time.perf_counter()

    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    if args.store_chunks:
        chunks_path = out_dir / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(chunks):
                f.write(json.dumps({"i": i, "text": text}, ensure_ascii=False) + "\n")

    write_time = time.perf_counter() - t_write

    total_time = time.perf_counter() - t_total

    stats = {
        "model": args.model,
        "embedding_dim": dim,
        "num_files": len(files),
        "num_chunks": len(chunks),

        # Тайминги
        "elapsed_chunking_s": round(chunk_time, 3),
        "elapsed_model_load_s": round(model_time, 3),
        "elapsed_embedding_s": round(emb_time, 3),
        "elapsed_faiss_io_s": round(faiss_time, 3),
        "elapsed_write_meta_s": round(write_time, 3),
        "elapsed_total_s": round(total_time, 3),
    }

    with open(out_dir / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
