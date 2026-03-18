"""Generate markdown results tables from evaluation results."""

from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from src.eval.harness import EvalResult


def generate_summary_table(results: list[EvalResult]) -> Table:
    """Create a rich table summarizing results by policy."""
    # Group by policy
    by_policy: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_policy[r.policy_name].append(r)

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Answer Acc", justify="right")
    table.add_column("Cit Prec", justify="right")
    table.add_column("Cit Recall", justify="right")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Avg Time", justify="right")

    for policy_name, policy_results in by_policy.items():
        n = len(policy_results)
        avg_reward = sum(r.reward for r in policy_results) / n
        avg_answer = sum(r.answer_score for r in policy_results) / n
        avg_cit_p = sum(r.citation_precision for r in policy_results) / n
        avg_cit_r = sum(r.citation_recall for r in policy_results) / n
        avg_steps = sum(r.steps for r in policy_results) / n
        avg_time = sum(r.duration_seconds for r in policy_results) / n

        table.add_row(
            policy_name,
            f"{avg_reward:.3f}",
            f"{avg_answer:.3f}",
            f"{avg_cit_p:.3f}",
            f"{avg_cit_r:.3f}",
            f"{avg_steps:.1f}",
            f"{avg_time:.1f}s",
        )

    return table


def generate_detail_table(results: list[EvalResult]) -> Table:
    """Create a per-question detail table."""
    table = Table(title="Per-Question Results", show_lines=True)
    table.add_column("Question", style="bold", max_width=40)
    table.add_column("Policy", style="cyan")
    table.add_column("Reward", justify="right")
    table.add_column("Answer", justify="right")
    table.add_column("Cit P", justify="right")
    table.add_column("Cit R", justify="right")
    table.add_column("Steps", justify="right")

    for r in sorted(results, key=lambda x: (x.question_id, x.policy_name)):
        table.add_row(
            f"{r.question_id}: {r.question[:35]}...",
            r.policy_name,
            f"{r.reward:.3f}",
            f"{r.answer_score:.3f}",
            f"{r.citation_precision:.3f}",
            f"{r.citation_recall:.3f}",
            str(r.steps),
        )

    return table


def generate_markdown(results: list[EvalResult]) -> str:
    """Generate markdown report text."""
    by_policy: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_policy[r.policy_name].append(r)

    lines = ["# Evaluation Results\n"]
    lines.append("| Policy | Avg Reward | Answer Acc | Cit Prec | Cit Recall | Avg Steps | Avg Time |")
    lines.append("|--------|-----------|-----------|---------|-----------|----------|---------|")

    for policy_name, policy_results in by_policy.items():
        n = len(policy_results)
        avg_r = sum(r.reward for r in policy_results) / n
        avg_a = sum(r.answer_score for r in policy_results) / n
        avg_cp = sum(r.citation_precision for r in policy_results) / n
        avg_cr = sum(r.citation_recall for r in policy_results) / n
        avg_s = sum(r.steps for r in policy_results) / n
        avg_t = sum(r.duration_seconds for r in policy_results) / n
        lines.append(
            f"| {policy_name} | {avg_r:.3f} | {avg_a:.3f} | "
            f"{avg_cp:.3f} | {avg_cr:.3f} | {avg_s:.1f} | {avg_t:.1f}s |"
        )

    return "\n".join(lines)


def print_results(results: list[EvalResult], verbose: bool = False) -> None:
    """Print results using rich tables."""
    console = Console()
    console.print(generate_summary_table(results))

    if verbose:
        console.print()
        console.print(generate_detail_table(results))

        # Print trajectories
        for r in results:
            if r.trajectory:
                console.print(f"\n[bold]{r.policy_name} — {r.question_id}[/bold]")
                for step in r.trajectory:
                    console.print(f"  Step {step.step}: {step.action[:80]}...")
                    console.print(f"    → {step.observation[:120]}...")
