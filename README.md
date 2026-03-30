# docpilot

Ask your docs anything from your terminal using a local RAG pipeline.

## What it does

You point docpilot at a documentation website. It scrapes the pages, breaks the text into chunks, converts each chunk into a vector embedding using Ollama, and stores everything in ChromaDB. When you ask a question, it finds the most semantically relevant chunks and feeds them to a local LLM to generate an answer.

No cloud. No API keys. Runs entirely on your machine.

## How it works (step by step)

```
Docs URL → Scraper → Chunker → Ollama Embeddings → ChromaDB
                                                         ↓
Your Question → Embed Question → Retrieve Top-K Chunks → LLM → Answer
```

1. **Scrape** — crawls the site concurrently, respects `--max-pages`
2. **Chunk** — splits text into smaller pieces so embeddings stay focused
3. **Embed** — converts each chunk to a vector using an Ollama embedding model
4. **Store** — saves vectors + original text in ChromaDB at `~/.docpilot/chroma`
5. **Retrieve** — at query time, embeds your question and finds the closest chunks
6. **Answer** — sends retrieved chunks as context to a local LLM

## Why these design choices

**Why local-first?**
No external API dependency, no cost per query, and your docs stay on your machine.

**Why vector search instead of keyword search?**
Semantic search finds relevant chunks even when the exact words don't match. "How do I make a venv" still finds chunks about `python -m venv`.

**Why chunking?**
LLMs have context limits. Chunking keeps each piece small enough to embed accurately and fit into the prompt without overflow.

**Why ChromaDB?**
Handles both vector storage and similarity search in one local library. No separate database server needed.

**Why Ollama?**
Runs models locally. Both the embedding model and the chat model run on your machine via Ollama.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) installed and running
- A chat model and an embedding model pulled in Ollama

```bash
ollama pull qwen2.5:latest
ollama pull mxbai-embed-large:335m
```

## Installation

```bash
# Recommended — installs as a global CLI tool
uv tool install .

# Development — editable install, picks up code changes with uv run
pip install -e .
```

## First run

On first run, docpilot creates `~/.docpilot/config.toml` with defaults:

```toml
default_model = "qwen2.5:latest"
default_embed_model = "mxbai-embed-large:335m"
db_path = "~/.docpilot/chroma"
```

You can change these via CLI without editing the file directly.

## Usage

```bash
# See available Ollama models
docpilot model list

# Set which model to use for chat and embeddings
docpilot model set qwen2.5:latest
docpilot model setembed mxbai-embed-large:335m

# Ingest a docs site
docpilot ingest "https://docs.python.org/3/" --workers 24 --max-pages 100 --batch-size 64

# Ask a question
docpilot ask "How do I create a virtual environment in Python?"
```

## Trade-offs worth knowing

**Chunking size**
Smaller chunks = better retrieval precision but may lose surrounding context. Larger chunks = more context but embeddings get noisy.

**Top-K retrieval**
Higher K means more chunks sent to the LLM = better recall but larger prompt and slower response. Lower K is faster but may miss relevant info.

**Concurrency (`--workers`)**
More workers = faster scraping but higher load on the target server and your machine. Default 16 is a reasonable middle ground.

**Embedding overflow**
If a chunk is too long for the embedding model's context window, docpilot automatically splits it and retries rather than failing the whole ingest.

## Troubleshooting

**`No such option: --workers`** — package is stale, reinstall:
```bash
uv tool install . --force
```

**Ollama model not found** — run `docpilot model list` and set an available model.

**Empty answers after ingest** — ingest may have failed silently. Re-run ingest and check for errors.

## License

MIT