# RAG-бот (локальная LLM через Ollama)

## Требования
- Добавьте файл rag_bot.py в папку из Задания 3
- Установленный Ollama и модель (пример):
  ```bash
  brew install ollama
  ollama pull llama3.2:3b
  ```

## Установка зависимостей
```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

## REPL
```bash
  python rag_bot.py --repl --index_dir index --alpha 0.3 --topn 5 --k 100
```

## REST API
```bash
uvicorn rag.api:app --reload --port 8000
```
POST /ask:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "query": "кто такая Мерилла",
  "k": 5,
  "llm_model": "llama3.2:3b"
}'
```
