# docpilot 🧭

> **Ask your docs anything. Get answers, not links.**

docpilot is a local-first, offline documentation intelligence CLI. Point it at any documentation — websites, PDFs, GitHub repos, man pages — and ask questions in plain English from your terminal. No cloud. No API keys. No hallucinations without sources.

---

## The Problem

Developers today juggle dozens of tools, each with sprawling documentation spread across different websites, formats, and versions. Current solutions fall short:

- **Search is keyword-based** — you get a list of pages, not an answer
- **Multiple doc sources** have no unified interface
- **Dense reference docs** require deep reading just to find one fact
- **Cloud AI tools** send your queries to third-party servers and hallucinate without grounding
- **Docs go stale** — no way to know if what you're reading still reflects the actual code

docpilot fixes all of this.

---

## Features

- 🔍 **Semantic search** across all your indexed documentation
- 🗂️ **Multi-source querying** — ask across multiple doc sets simultaneously
- 🤖 **Local LLM answers** powered by Ollama (Mistral, LLaMA 3, Gemma, etc.)
- 📄 **Source citations** — every answer tells you exactly where it came from
- 🔄 **Staleness detection** — flags docs that may be out of sync with the source repo
- 🔒 **Fully offline** — your queries never leave your machine
- ⚡ **Fast retrieval** via ChromaDB vector storage


## License

MIT License — see [LICENSE](./LICENSE) for details.
