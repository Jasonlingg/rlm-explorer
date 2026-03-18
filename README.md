# RLM Explorer

An RL environment for training language models to actively explore document collections via code execution in a persistent REPL, rather than passively consuming retrieved context.

## Why This Matters

There's a fundamental difference between a researcher who knows how to use a library — searching, cross-referencing, computing, iterating — and one who reads whatever you put on their desk. Standard RAG is the latter: retrieve top-k chunks, stuff them into a prompt, hope for the best. RLM Explorer is the former: the agent writes Python code to search, read, extract, and compute across documents, iterating until it's confident in its answer.

Research shows iterative exploration outperforms single-pass RAG by up to +25 percentage points on multi-hop questions, even when the single-pass system is given perfect oracle context.

## The Core Loop

```
env.reset()          →  Agent receives question + tool descriptions
env.step(code)       →  Agent writes Python, observes stdout/stderr
env.step(code)       →  Agent refines search, cross-references docs
env.step(code)       →  Agent computes aggregations, verifies findings
env.step(SUBMIT)     →  Agent submits answer + citations → receives reward
```

Each episode spawns a persistent Python session. Variables, DataFrames, and helper functions survive across steps — like a Jupyter notebook the agent controls. The reward signal is verifiable: token overlap F1 for answer accuracy, precision/recall for citations, and an efficiency bonus.

## Architecture

```
src/
├── env/
│   ├── document_env.py   # Gym-compatible environment: reset(), step(), reward()
│   ├── repl.py           # Persistent REPL: Docker sandbox + local fallback
│   ├── corpus.py         # Load docs, chunk, embed, FAISS index
│   ├── reward.py         # Verifiable reward: answer F1, citation P/R, efficiency
│   └── tools.py          # Tool preamble: search(), read(), extract(), aggregate()
├── policies/
│   ├── claude_policy.py  # Reference policy: Claude explores iteratively
│   ├── naive_rag.py      # Baseline: top-k retrieve → answer in 1 step
│   ├── stuffing.py       # Baseline: concatenate all docs → answer in 1 step
│   └── single_shot.py    # Baseline: minimal retrieval → answer in 1 step
└── eval/
    ├── harness.py        # Run policies through env, collect trajectories
    ├── scorer.py         # Scoring functions
    └── report.py         # Results tables
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/jasonlingg/rlm-explorer.git
cd rlm-explorer
pip install -e ".[dev]"

# Generate synthetic corpus (28 cross-referencing business documents)
python scripts/setup_corpus.py

# Run tests
pytest tests/ -v

# Run Claude policy on one question
python scripts/run_eval.py --policy claude_policy --question q01 --verbose

# Run full evaluation (all policies, all 18 questions)
python scripts/run_eval.py
```

## Reward Signal

The reward is designed for GRPO training:

```
reward = 0.5 × answer_F1 + 0.25 × citation_precision + 0.25 × citation_recall + efficiency_bonus
```

- **Answer F1**: Token overlap between predicted and gold answer
- **Citation Precision**: Fraction of cited documents that are correct
- **Citation Recall**: Fraction of required documents that were cited
- **Efficiency Bonus**: Up to 0.2 extra for solving in fewer steps

## Question Types

The evaluation set contains 18 questions across 5 types:
- **Cross-document aggregation**: Combine numbers from multiple documents
- **Cross-document comparison**: Compare metrics or clauses across documents
- **Multi-hop reasoning**: Find X in doc A → look up related info in doc B
- **Single-document extraction**: Straightforward lookups (baseline sanity check)
- **Contradiction detection**: Identify inconsistencies across documents

## Future Work

Plug in an open-weight model (Llama, Qwen) with GRPO on Prime Intellect to train a model that learns exploration behavior from this environment's reward signal.

## Built With

- [Claude API](https://docs.anthropic.com) — Reference policy
- [sentence-transformers](https://sbert.net) — Document embeddings (all-MiniLM-L6-v2)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [Docker](https://www.docker.com) — Sandboxed code execution

## Development Stack

- [Claude Code](https://claude.ai/claude-code) — AI-assisted development, debugging, and implementation
- [Google NotebookLM](https://notebooklm.google.com) — Research synthesis and project documentation. The notebook aggregates multi-hop RAG papers, codebase source files, and eval run logs into a single queryable knowledge base for iterating on environment design. Used agentic skill for Google NotebookLM.

## License

MIT
