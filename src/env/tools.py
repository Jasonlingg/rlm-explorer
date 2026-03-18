"""Tool preamble injected into the REPL at session start.

Defines search(), read(), extract(), aggregate() functions that operate on the
mounted corpus directory. Includes memoization so re-running the cumulative
script doesn't repeat expensive file reads.
"""

TOOL_PREAMBLE = '''
import json
import os
import re
from pathlib import Path

# Memoization cache — persists across cumulative script re-runs
if "_memo" not in dir():
    _memo = {}

CORPUS_DIR = Path("/workspace/corpus") if Path("/workspace/corpus").exists() else Path("data/corpus")

def _load_doc(doc_id: str) -> dict | None:
    """Load a document by ID, with memoization."""
    if doc_id in _memo:
        return _memo[doc_id]
    path = CORPUS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        doc = json.load(f)
    _memo[doc_id] = doc
    return doc

def _load_all_docs() -> list[dict]:
    """Load all documents, with memoization."""
    if "_all_docs" in _memo:
        return _memo["_all_docs"]
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        with open(path) as f:
            docs.append(json.load(f))
    _memo["_all_docs"] = docs
    return docs

def search(query: str, top_k: int = 5) -> list[dict]:
    """Keyword search over corpus files. Returns [{doc_id, chunk, score}]."""
    query_terms = query.lower().split()
    results = []
    for doc in _load_all_docs():
        text = doc["text"].lower()
        score = sum(text.count(term) for term in query_terms)
        if score > 0:
            # Return first 500 chars as chunk preview
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title", doc["doc_id"]),
                "chunk": doc["text"][:500],
                "score": score,
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def read(doc_id: str) -> str:
    """Read full document text by ID."""
    doc = _load_doc(doc_id)
    if doc is None:
        return f"ERROR: Document '{doc_id}' not found"
    return doc["text"]

def extract(doc_id: str, pattern: str) -> list[str]:
    """Extract text matching a regex pattern from a document."""
    doc = _load_doc(doc_id)
    if doc is None:
        return [f"ERROR: Document '{doc_id}' not found"]
    return re.findall(pattern, doc["text"])

def aggregate(doc_ids: list[str], field: str) -> list[dict]:
    """Extract a JSON metadata field across multiple documents."""
    results = []
    for doc_id in doc_ids:
        doc = _load_doc(doc_id)
        if doc is None:
            results.append({"doc_id": doc_id, "error": "not found"})
            continue
        value = doc.get("metadata", {}).get(field)
        results.append({"doc_id": doc_id, field: value})
    return results

def list_docs() -> list[dict]:
    """List all available documents."""
    return [
        {"doc_id": d["doc_id"], "title": d.get("title", d["doc_id"]), "chars": len(d["text"])}
        for d in _load_all_docs()
    ]

print("Tools loaded: search(), read(), extract(), aggregate(), list_docs()")
print(f"Corpus: {len(list_docs())} documents available")
'''
