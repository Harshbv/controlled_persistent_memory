# Controlled Persistent Memory for LLM Assistants

LLM assistants that remember users usually store *everything* and retrieve by similarity, the "vanilla RAG" approach. That bloats memory, and it brings back stale preferences: the user said they like tea in turn 3 and switched to coffee in turn 40, yet both get retrieved.

This project replaces store-everything with a **controlled** memory layer that decides what is worth writing, versions memories when preferences change, and retrieves only what is still true.

> Research project for **CSE 572 (Data Mining), Arizona State University, 2026**. Evaluated on the **PersonaMem v1 (32k)** benchmark.

---

## Results

On the full **588-episode** evaluation, the best controlled-memory variant reached **66% accuracy** while storing **~60% fewer memories** than vanilla RAG.

On the 100-episode dev split (`metrics_100ep_final.txt`):

| Method | Accuracy | Avg. memories stored / episode |
|---|---|---|
| No memory | 43% | 0 |
| Summary only | 42% | 0 |
| Vanilla RAG | 69% | 177.3 |
| **Controlled (ours)** | **72%** | **49.9** |

The controlled method matched vanilla RAG's accuracy using roughly a quarter of its memory, and it retired an average of 7.8 outdated memories per episode through versioning.

---

## How it works

```
User turns ──► Write router ──► Versioning ──► Chroma store (active / superseded)
                (what to keep)   (supersede old            │
                                  conflicting memories)    ▼
Question ──────────────────────────────────────► Retrieve top-k ACTIVE memories ──► Llama 3.3 70B (Groq)
```

- **Write router** (`memory/write_router_v1.py`): pattern-based extraction of preference statements (*prefer, like, love, don't like, favorite, from now on…*), normalized to canonical forms such as "I don't like phone calls" → `dislikes phone calls`.
- **Versioning** (`memory/versioning.py`): when a new memory conflicts with an older one for the same persona and memory type, the old one is marked `superseded` and linked rather than deleted, so the full history stays auditable.
- **Retrieval**: only `active` memories for the persona are eligible. Dense (MiniLM) similarity is combined with BM25, option-conditioned and temporal signals, then deduplicated.
- **Logging**: per-episode JSONL logs of every write, supersession and retrieval, for debugging and analysis.

See [`memory/README.md`](memory/README.md) for the data model and the write and retrieval paths in detail.

---

## Repository layout

| Path | Purpose |
|---|---|
| `memory/` | Controlled memory subsystem: schema, write router, versioning, Chroma store, logging |
| `baselines/` | Vanilla RAG, summary-only and embedding baselines |
| `eval/` | Evaluation runner, per-method adapters, scoring and metrics |
| `data/` | PersonaMem loading and episode construction |
| `configs/default.yaml` | Shared config so every method is compared under the same settings |
| `metrics_*.txt` | Result tables with per-topic and per-question-type breakdowns |

---

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key"

# Build evaluation episodes from PersonaMem
python -m data.build_episode

# Run a method: no_memory | summary_only | vanilla_rag | controlled
python -m eval.run_eval --method controlled --episodes_path data/dev_100.jsonl

# Sanity-check supersession on its own
python -m memory.demo_versioning
```

**Stack:** Python · ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · Groq (Llama 3.3 70B) · PersonaMem

---

## Team

Built with [@vnaruvan](https://github.com/vnaruvan), [@AravindanC7](https://github.com/AravindanC7), [@rajius2024](https://github.com/rajius2024) and [@Harshbv](https://github.com/Harshbv). The original repository is [rajius2024/controlled_persistent_memory](https://github.com/rajius2024/controlled_persistent_memory).
