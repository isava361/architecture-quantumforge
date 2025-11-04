# Векторный индекс базы знаний (FAISS + all-MiniLM-L6-v2)

**Модель:** all-MiniLM-L6-v2 (384 dim) — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Как запустить

Создать папку ./data/ и перенести в нее файлы .md из Задания 2.

Установка:
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Индексация:
```bash
python3 build_index.py --data_dir data --out_dir index \
  --use_lc_splitter \
  --lc_chunk_chars 1200 \
  --lc_chunk_overlap_chars 200 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --store_chunks
```

Поиск:
```bash
python3 query_index.py --query "кто такая мирелла" \
  --k 50 --topn 5 \
  --hybrid_bm25 --alpha 0.7 \
  --rerank_cross_encoder cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Детали реализации (по пунктам задания)

Эмбеддинг-модель: all-MiniLM-L6-v2
• Репозиторий/API: SentenceTransformers (Hugging Face)
• Размер эмбеддинга: 384
• В коде: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
• Нормализация включена → косинусная близость через FAISS IndexFlatIP.

Чанкинг:

По умолчанию — 200 слов с overlap 40 (100–300 слов — в допусках).

Метаданные каждого чанка содержат: path, title, chunk_index, id, примерную длину и source.

Генерация эмбеддингов:

Батч-код через SentenceTransformers, normalize_embeddings=True.

Хранятся только в FAISS; метаданные — в metadata.jsonl (строго в порядке добавления).

Векторная БД (FAISS):

Тип: IndexFlatIP (cosine/IP на нормализованных векторах).

Файлы: index/faiss.index, index/metadata.jsonl, index/build_stats.json.

build_stats.json фиксирует: модель, размерность, сколько чанков и время генерации
