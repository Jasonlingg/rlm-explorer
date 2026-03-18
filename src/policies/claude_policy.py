"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import anthropic
from loguru import logger

SYSTEM_PROMPT = """You are an expert data analyst exploring a document corpus through a Python REPL.

Your environment has these tools already loaded:
- search(query, top_k=5) → keyword search over documents, returns [{doc_id, title, chunk, score}]
- read(doc_id) → returns full document text
- extract(doc_id, pattern) → regex extraction from a document
- aggregate(doc_ids, field) → extract a metadata field across multiple documents
- list_docs() → list all available documents with {doc_id, title, chars}

You also have pandas, numpy, json, re available.

Strategy:
1. Start by searching for relevant documents
2. Read promising documents in full
3. Cross-reference information across documents
4. Compute or aggregate as needed
5. When confident, submit your answer

IMPORTANT: Each response must be EITHER:
- Python code to execute (will be run in the REPL, output returned to you)
- A submission in this exact format:
  SUBMIT: <your detailed answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Write clean, focused code. Print results you want to see. Be thorough but efficient."""


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        """Given an observation, return an action (code or SUBMIT)."""
        self.history.append({"role": "user", "content": observation})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )

        action = response.content[0].text
        self.history.append({"role": "assistant", "content": action})

        # Strip markdown code fences if present
        action = self._strip_code_fences(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```python ... ``` wrappers if the entire response is a code block."""
        stripped = text.strip()
        if stripped.startswith("```python") and stripped.endswith("```"):
            return stripped[len("```python"):][:-3].strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            # Remove first line (```lang) and last line (```)
            lines = stripped.split("\n")
            return "\n".join(lines[1:-1]).strip()
        return text
