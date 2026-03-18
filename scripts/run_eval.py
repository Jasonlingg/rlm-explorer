"""CLI for running evaluation: policies through the document exploration environment."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from loguru import logger
from rich.console import Console

from src.env.corpus import Corpus
from src.eval.harness import run_eval
from src.eval.report import print_results
from src.policies.claude_policy import ClaudePolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.stuffing import ContextStuffingPolicy

app = typer.Typer(help="RLM Explorer Evaluation CLI")
console = Console()


def load_questions(path: str = "data/questions/eval_set.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_policies(
    corpus: Corpus,
    policy_names: list[str] | None = None,
) -> dict[str, object]:
    """Build policy instances. If policy_names is None, build all."""
    all_policies = {
        "claude_policy": lambda: ClaudePolicy(),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus),
    }

    if policy_names:
        return {
            name: factory()
            for name, factory in all_policies.items()
            if name in policy_names
        }
    return {name: factory() for name, factory in all_policies.items()}


@app.command()
def main(
    policy: str | None = typer.Option(
        None, "--policy", "-p", help="Run only this policy (e.g. claude_policy)"
    ),
    question: str | None = typer.Option(
        None, "--question", "-q", help="Run only this question ID (e.g. q01)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print full trajectories"
    ),
    max_steps: int = typer.Option(
        10, "--max-steps", help="Max steps per episode"
    ),
    questions_path: str = typer.Option(
        "data/questions/eval_set.json", "--questions", help="Path to questions JSON"
    ),
    corpus_path: str = typer.Option(
        "data/corpus", "--corpus", help="Path to corpus directory"
    ),
) -> None:
    """Run evaluation: policies through the document exploration environment."""
    console.print("[bold]RLM Explorer — Evaluation[/bold]\n")

    # Load corpus
    console.print("Loading corpus...")
    corpus = Corpus(corpus_path=corpus_path)
    corpus.load()

    # Load questions
    questions = load_questions(questions_path)
    console.print(f"Loaded {len(questions)} questions\n")

    # Build policies
    policy_names = [policy] if policy else None
    policies = build_policies(corpus, policy_names)
    console.print(f"Policies: {', '.join(policies.keys())}\n")

    # Filter questions
    question_ids = [question] if question else None

    # Run evaluation
    results = run_eval(
        corpus=corpus,
        questions=questions,
        policies=policies,
        max_steps=max_steps,
        use_docker=False,
        corpus_path=corpus_path,
        question_ids=question_ids,
    )

    # Print results
    console.print()
    print_results(results, verbose=verbose)


if __name__ == "__main__":
    app()
