# Jarvis Lite

A private, local-first AI assistant that answers questions from your own Markdown / Obsidian notes using retrieval-augmented generation (RAG).

Everything runs on your machine. Your notes, their embeddings and the language model never leave it.

## How it works

```
Markdown / Obsidian vault
        │
        ▼
knowledge.py        load every .md note (skips .obsidian and instruction files)
        │
        ▼
retrieval.py        Markdown-aware chunking → EmbeddingGemma embeddings
        │           → persistent ChromaDB index (cosine similarity)
        ▼
search_index        top-k semantic search + relevance threshold
        │
        ▼
expand_sections     add missing sibling sections for "whole system" questions
        │
        ▼
assistant.py        grounded system prompt + retrieved context → local LLM via Ollama
```

## What makes the retrieval interesting

**1. Markdown-aware chunking.** Notes are split along their heading hierarchy instead of every N words. Each section stays whole, only sections over 350 words are split, and every chunk carries its heading path (for example `Weekly Plan > Monday`). That path is both embedded with the text and stored as metadata.

**2. Correct similarity metric.** The Chroma collection is created with `hnsw:space: cosine`. Computing `1 - distance` on Chroma's default L2 distance produced negative "similarity" scores. Switching the metric fixed the ranking.

**3. Semantic similarity is not completeness.** Ask *"what is my weekly plan?"* and top-5 search may return `Saturday` and `Wednesday` and silently drop the other five days, because sibling sections look almost identical to an embedding model. `expand_sections` fixes this with a small, explainable rule:

- If the question names one of the sibling headings, it was a specific question, so nothing is expanded.
- If it names none of them, it was a question about the whole group, so all siblings are added in document order.

It is plain string matching on heading names, not a classifier, so its behaviour can be read and predicted.

**4. Grounded answers.** The system prompt treats retrieved notes as *data, not instructions*. It tells the model not to invent personal facts and to label anything beyond the notes as an inference.

**5. Incremental index.** Chunks are upserted with stable IDs (`path::chunk_index`), and chunks from deleted notes are removed on the next run.

## Setup

Requirements: Python 3.12, [Ollama](https://ollama.com), and a folder of Markdown notes.

```bash
# 1. Pull the models
ollama pull embeddinggemma
ollama pull deepseek-r1:7b        # or any chat model you prefer

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 3. Configure
copy config\config.example.json config\config.json   # macOS/Linux: cp
#    then edit config.json: set vault_path and instructions_path

# 4. Run from the project root
python app/assistant.py
```

Type `exit` to quit. The first run builds the index in `data/chroma_db/`.

## Configuration

| Key | Meaning |
|---|---|
| `model` | Ollama chat model used to answer |
| `vault_path` | Folder containing your Markdown notes |
| `instructions_path` | Markdown file with the assistant's core instructions (excluded from retrieval) |

## Project structure

```
app/
  assistant.py        chat loop, context building, grounded prompt
  retrieval.py        chunking, indexing, search, section expansion
  knowledge.py        Markdown loader
  embeddings_test.py  small cosine-similarity experiment with EmbeddingGemma
config/
  config.example.json
```

## Roadmap

- REST API with FastAPI, plus tests
- Docker image and cloud deployment
- Agentic tools with LangGraph (reminders, file operations)
- Voice input and output

## Tech

Python · Ollama · EmbeddingGemma · ChromaDB · RAG
