"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import re

import anthropic
from loguru import logger

SYSTEM_PROMPT = """You are a Python programmer exploring a document corpus. You can ONLY respond with Python code or a SUBMIT line. Never respond with English prose, explanations, or XML.

AVAILABLE FUNCTIONS (already loaded):
  search(query, top_k=5) → returns list of dicts: [{"doc_id": "...", "title": "...", "chunk": "...", "score": N}]
  read(doc_id) → returns full document text as string
  extract(doc_id, pattern) → returns list of regex matches
  list_docs() → returns [{"doc_id": "...", "title": "...", "chars": N}]

RULES:
1. Each response must be EITHER executable Python code OR a SUBMIT line. Never both.
2. Do NOT write English sentences. Do NOT explain your thinking. Just write code.
3. Do NOT use XML tags, markdown fences, or any non-Python syntax.
4. Use print() to see results. Variables persist between steps.
5. When you have the answer, respond with ONLY: SUBMIT: <answer> CITATIONS: ["id1", "id2"]

EXAMPLE STEP 1 — search and print:
results = search("net income 2024")
for r in results:
    print(r["doc_id"], r["title"], r["score"])

EXAMPLE STEP 2 — read a document:
text = read("apex_corp_2024_financial")
print(text)

EXAMPLE STEP 3 — extract data:
matches = extract("apex_corp_2024_financial", r"net income.*?\\$([\d,.]+)")
print(matches)

EXAMPLE STEP 4 — submit answer:
SUBMIT: The net income was $28.3M. CITATIONS: ["apex_corp_2024_financial"]"""


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
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

        # Clean raw model output into executable Python
        action = self._clean_action(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []

    @staticmethod
    def _clean_action(text: str) -> str:
        """Extract executable Python from model output, stripping fences, XML, and prose."""
        stripped = text.strip()

        # If there's a SUBMIT line anywhere, extract and return it
        # (model is ready to answer — don't try to run code too)
        submit_match = re.search(
            r"(SUBMIT:\s*.*?CITATIONS:\s*\[.*?\])",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if not submit_match:
            submit_match = re.search(
                r"(SUBMIT:\s*.+)",
                stripped,
                re.IGNORECASE,
            )
        if submit_match:
            return submit_match.group(1).strip()

        # Strip markdown code fences
        if stripped.startswith("```python") and stripped.endswith("```"):
            return stripped[len("```python"):][:-3].strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.split("\n")
            return "\n".join(lines[1:-1]).strip()

        # Strip XML function_calls (Haiku sometimes generates these)
        if "<function_calls>" in stripped or "<invoke" in stripped:
            stripped = re.sub(r"</?function_calls>", "", stripped)
            stripped = re.sub(r"</?invoke[^>]*>", "", stripped)
            stripped = re.sub(r"</?parameter[^>]*>", "", stripped)

        # Extract code from markdown fences embedded in prose
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", stripped, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks).strip()

        # Filter: keep only lines that look like Python code, drop prose
        lines = stripped.split("\n")
        code_lines = []
        for line in lines:
            s = line.strip()
            # Keep blank lines (they're valid Python)
            if not s:
                code_lines.append(line)
                continue
            # Keep lines that look like code
            if re.match(
                r"^("
                r"#|"                           # comments
                r"[a-zA-Z_]\w*\s*[=(.\[]|"      # assignment, call, attribute, index
                r"from |import |"               # imports
                r"print\(|"                     # print
                r"for |if |elif |else:|while |" # control flow
                r"def |class |"                 # definitions
                r"return |yield |"              # returns
                r"try:|except |finally:|"       # exception handling
                r"with |"                       # context managers
                r"raise |assert |"              # raise/assert
                r"pass|break|continue|"         # simple statements
                r"\)|"                          # closing paren (continuation)
                r"\]|"                          # closing bracket
                r"\}|"                          # closing brace
                r"\"\"\"|'''|"                  # docstrings
                r"@"                            # decorators
                r")",
                s,
            ):
                code_lines.append(line)
            # else: drop the line (it's prose)

        result = "\n".join(code_lines).strip()
        return result if result else stripped
