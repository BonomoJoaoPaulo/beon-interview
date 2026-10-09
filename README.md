# AI Interview API

API FastAPI mínima para live coding: `GET /health` e `POST /ask` (pergunta respondida pelo Ollama `llama3.2:1b` usando `knowledge_base/beon_info.txt`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha as chaves, se for usar LLM
```

## Rodar

```bash
uvicorn app.main:app --reload
# docs interativas: http://localhost:8000/docs
# requer `ollama serve` rodando e `ollama pull llama3.2:1b`
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What does BEON.tech do?"}'
# -> {"answer":"...","in_scope":true}
```

## Testes

```bash
pytest
```
