# docpilot

Local-first CLI for document ingestion and question answering with Ollama + Chroma.

docpilot is a FOSS Hack project focused on practical developer workflows: ingest docs quickly, query them from terminal, and keep everything on-device.

## Highlights

- Local RAG pipeline (no cloud APIs required)
- Website and sitemap ingestion
- Local file ingestion for PDF and CSV
- Concurrent crawling and threaded chunk preparation
- Configurable speed profiles for faster or higher-quality answers
- Simple CLI designed for fast iteration

## Architecture

```text
Source (URL / Sitemap / PDF / CSV)
    -> Extract text
    -> Chunk
    -> Embed (Ollama embeddings)
    -> Store vectors (Chroma)

Question
    -> Retrieve top matches
    -> Compose context
    -> Generate answer (Ollama chat model)
```

## Requirements

- Python 3.12+
- Ollama installed and running: https://ollama.com/
- At least one chat model and one embedding model pulled locally

```bash
ollama pull qwen2.5:latest
ollama pull mxbai-embed-large:335m
```

## Installation

```bash
# Standard install
uv tool install .

# Development mode (recommended while iterating)
uv tool install --editable .
```

## Quick Start

```bash
# 1) Pick models
docpilot model list
docpilot model set qwen2.5:latest
docpilot model setembed mxbai-embed-large:335m

# 2) Optional: set response profile
docpilot speed fast

# 3) Ingest content
docpilot ingest "https://docs.python.org/3/" --max-pages 100 --workers 24

# 4) Ask questions
docpilot ask "How do I create a virtual environment?"
```

## CLI Reference

### ingest

Ingest content into the local vector database.

```bash
docpilot ingest SOURCE [OPTIONS]
```

`SOURCE` can be:

- Website URL (for crawl): `https://example.com/docs`
- Sitemap URL: `https://example.com/sitemap.xml`
- Local PDF file: `./docs/guide.pdf`
- Local CSV file: `./data/faq.csv`

Options:

- `--max-pages`, `-p` (default: `20`): max pages to crawl for websites
- `--workers`, `-w` (default: `16`): concurrent crawl workers
- `--batch-size`, `-b` (default: `32`): embedding batch size
- `--embed-workers`, `-e` (default: `0`): threads for chunk preparation (`0` = auto)

Examples:

```bash
docpilot ingest "https://docs.python.org/3/" --max-pages 100 --workers 24 --batch-size 64
docpilot ingest "./docs/guide.pdf"
docpilot ingest "./data/reviews.csv"
```

### ask

Query the ingested knowledge base.

```bash
docpilot ask "What is MongoDB used for?"
```

### model

Manage Ollama model selection.

```bash
docpilot model list
docpilot model set <chat-model>
docpilot model setembed <embedding-model>
```

### speed

Set answer-generation profile.

```bash
docpilot speed [fast|balanced|quality]
```

- `fast`: lower latency, shorter context/output
- `balanced`: default trade-off
- `quality`: larger context/output, slower responses

## Configuration

Configuration is stored at `~/.docpilot/config.toml`.

Common keys include:

- `default_model`
- `default_embed_model`
- `db_path`
- `retrieval_k`
- `max_context_chars`
- `max_doc_chars`
- `num_predict`
- `num_ctx`
- `num_thread`
- `temperature`

## Performance Notes

- Use `docpilot speed fast` for lower latency answers.
- Increase `--workers` for faster crawling.
- Increase `--batch-size` when your machine can handle larger embedding batches.
- Keep `--embed-workers 0` to auto-tune chunking thread count.

## Troubleshooting

### HTTP 429 while crawling (rate limited)

- Use fewer workers for strict sites: `--workers 6` to `--workers 12`
- Keep `--max-pages` realistic for the target site
- Retry after a short cool-down if the site is actively throttling
- docpilot now retries with backoff and respects `Retry-After` when provided

Example:

```bash
docpilot ingest "https://example.com/docs" --max-pages 100 --workers 8
```

Note: the correct flag is `--max-pages` (not `--maxpages`).

### Command changes not showing

Reinstall in editable mode during development:

```bash
uv tool install --editable .
```

### Model not found

List local models and set valid names:

```bash
docpilot model list
```

### Empty or weak answers

- Ensure ingestion completed successfully
- Use `docpilot speed quality` for deeper context
- Re-ingest with higher page limits or better source coverage

## Contributing

Contributions are welcome. If you are using docpilot in FOSS Hack or related projects, open issues and PRs with clear reproduction steps and expected behavior.

## License

MIT