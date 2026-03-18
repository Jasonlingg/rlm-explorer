"""Context stuffing baseline: concatenate all documents, answer in 1 step.

Represents the "just give it everything" approach. Tests whether the model
can find needles in the haystack without targeted retrieval.
"""

from __future__ import annotations

import anthropic
from loguru import logger

from src.env.corpus import Corpus

STUFFING_PROMPT = """Answer the following question using ONLY the provided documents.
Include citations as document IDs.

Format your response exactly as:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Documents:
{documents}

Question: {question}"""


class ContextStuffingPolicy:
    """Baseline: concatenate all docs into context, answer in one step."""

    def __init__(
        self,
        corpus: Corpus,
        model: str = "claude-haiku-4-5-20251001",
        max_context_chars: int = 100_000,
    ) -> None:
        self.corpus = corpus
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_context_chars = max_context_chars
        self._question: str | None = None
        self._answered: bool = False

    def act(self, observation: str) -> str:
        """Extract question, stuff all docs into context, submit."""
        if self._answered:
            return 'SUBMIT: No answer available CITATIONS: []'

        if self._question is None:
            self._question = self._extract_question(observation)

        # Concatenate all documents
        doc_parts = []
        total_chars = 0
        for doc_info in self.corpus.list_documents():
            text = self.corpus.read(doc_info.doc_id)
            if text and total_chars + len(text) < self.max_context_chars:
                doc_parts.append(f"=== [{doc_info.doc_id}] {doc_info.title} ===\n{text}")
                total_chars += len(text)

        documents = "\n\n".join(doc_parts)

        prompt = STUFFING_PROMPT.format(documents=documents, question=self._question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        self._answered = True
        action = response.content[0].text.strip()
        logger.debug(f"ContextStuffing answer: {action[:100]}...")
        return action

    def reset(self) -> None:
        self._question = None
        self._answered = False

    @staticmethod
    def _extract_question(observation: str) -> str:
        for line in observation.split("\n"):
            if line.strip().startswith("Question:"):
                return line.strip()[len("Question:"):].strip()
        return observation.split("\n")[-1].strip()
