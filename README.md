# RadTherapy RAG Agent

A small multi-agent Retrieval-Augmented Generation (RAG) system for answering
educational questions about external beam radiotherapy — IMRT, inverse
planning, dose calculation, and dose-volume histograms (DVHs).

The project runs entirely locally (Ollama + ChromaDB) and demonstrates a
structured-output, multi-agent LangGraph pipeline: a router decides whether
retrieval is needed, a retriever node queries a local vector store, an answer
node generates a grounded response, and a critic node checks the answer
against the retrieved context before allowing it to be returned.

## Architecture

```
START
  │
  ▼
router  ──(needs_retrieval=False)──▶ answer ──▶ critic ──▶ END
  │
  (needs_retrieval=True)
  │
  ▼
retriever ──▶ answer ──▶ critic ──┬──(is_grounded=True)──▶ END
                                  └──(is_grounded=False, first try)──▶ increment_retry ──▶ answer
```

- **router** — classifies the question and decides if it needs retrieval
  from the local document corpus, using a structured (`RouterDecision`)
  LLM output.
- **retriever** — embeds the search query and queries a local ChromaDB
  collection for the most relevant chunks.
- **answer** — generates a grounded answer from the retrieved context only,
  citing which sources it relied on (`AnswerOutput`). Returns an
  "insufficient context" response if no relevant chunks were retrieved.
- **critic** — checks whether the answer is actually supported by the cited
  context (`CriticOutput`). If not grounded, the pipeline retries the answer
  step once with feedback before giving up.

All inter-node communication uses [Pydantic](https://docs.pydantic.dev/)
schemas (`src/schemas.py`) as structured output, so each LLM response is
validated and typed rather than parsed from free text.

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) for the multi-agent
  state graph
- [Ollama](https://ollama.com/) running `llama3.1:8b` locally as the LLM
- [ChromaDB](https://www.trychroma.com/) as the local vector store
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for
  embeddings
- [Pydantic v2](https://docs.pydantic.dev/) for structured LLM outputs

## Project structure

```
src/
  config.py       # paths, model names, chunking/retrieval parameters
  ingest.py       # parses data/raw/*.txt, chunks it, embeds it into ChromaDB
  schemas.py      # Pydantic models used as structured LLM output
  graph.py        # LangGraph nodes, routing logic, and graph assembly
demo.py           # interactive CLI demo
tests/            # unit tests for parsing, chunking, and retry/routing logic
data/raw/         # small set of IAEA-based educational source documents
data/chroma_db/   # local vector store (generated, not committed)
```

## Setup

Requires Python 3.9+ and [Ollama](https://ollama.com/) installed locally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1:8b
```

Copy `.env.example` to `.env` if you want to override any defaults (model
name, Ollama URL, collection name).

## Usage

**1. Build the vector store** from the documents in `data/raw/`:

```bash
python -m src.ingest --reset
```

**2. Run the interactive demo:**

```bash
python demo.py
```

Example question: `what is inverse planning in IMRT?`

**3. Run the tests:**

```bash
pytest tests/ -v
```

The tests only cover deterministic logic (document parsing, text chunking,
router/critic routing decisions) and do not require Ollama or ChromaDB to be
running.

## Design notes

- The critic can trigger **at most one retry** — this keeps latency bounded
  and avoids infinite loops on a local, resource-constrained setup.
- If the router decides retrieval isn't needed (e.g. small talk), the answer
  node still responds conservatively with "insufficient context" rather than
  answering from general knowledge. This is a deliberate choice: the system
  is scoped to only answer from its document corpus, which is a more
  appropriate default for a medical-adjacent domain.
- Source documents are short IAEA-based educational summaries, chosen to
  keep the corpus small and the pipeline easy to reason about end-to-end
  within a limited time budget.

## License

MIT — see [LICENSE](LICENSE).
