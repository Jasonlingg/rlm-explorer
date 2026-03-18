"""Evaluation harness: run policies through the environment and collect results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, StepRecord


@dataclass
class EvalResult:
    question_id: str
    question: str
    policy_name: str
    reward: float
    answer_score: float
    citation_precision: float
    citation_recall: float
    efficiency_bonus: float
    steps: int
    trajectory: list[StepRecord] = field(default_factory=list)
    predicted_answer: str = ""
    predicted_citations: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def run_single(
    env: DocumentExplorationEnv,
    policy: object,
    question_idx: int,
) -> EvalResult:
    """Run a single policy on a single question through the environment."""
    # Reset
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset(question_idx=question_idx)
    q = env.questions[question_idx % len(env.questions)]

    start = time.time()
    steps = 0
    reward = 0.0
    predicted_answer = ""
    predicted_citations: list[str] = []
    answer_score = 0.0
    cit_p = 0.0
    cit_r = 0.0
    eff = 0.0

    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        if done and "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            answer_score = rb.answer_score
            cit_p = rb.citation_precision
            cit_r = rb.citation_recall
            eff = rb.efficiency_bonus
            predicted_answer = info.get("predicted_answer", "")
            predicted_citations = info.get("predicted_citations", [])

    duration = time.time() - start
    trajectory = env.get_trajectory()

    return EvalResult(
        question_id=q["id"],
        question=q["question"],
        policy_name=type(policy).__name__,
        reward=reward,
        answer_score=answer_score,
        citation_precision=cit_p,
        citation_recall=cit_r,
        efficiency_bonus=eff,
        steps=steps,
        trajectory=trajectory,
        predicted_answer=predicted_answer,
        predicted_citations=predicted_citations,
        duration_seconds=duration,
    )


def run_eval(
    corpus: Corpus,
    questions: list[dict],
    policies: dict[str, object],
    max_steps: int = 10,
    use_docker: bool | None = None,
    corpus_path: str = "data/corpus",
    question_ids: list[str] | None = None,
) -> list[EvalResult]:
    """Run all policies on all (or selected) questions."""
    results: list[EvalResult] = []

    # Filter questions if specific IDs requested
    if question_ids:
        q_indices = [
            i for i, q in enumerate(questions) if q["id"] in question_ids
        ]
    else:
        q_indices = list(range(len(questions)))

    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=max_steps,
        use_docker=use_docker,
        corpus_path=corpus_path,
    )

    try:
        for policy_name, policy in policies.items():
            logger.info(f"Running policy: {policy_name}")

            for q_idx in q_indices:
                q = questions[q_idx]
                logger.info(f"  Question {q['id']}: {q['question'][:60]}...")

                try:
                    result = run_single(env, policy, q_idx)
                    result.policy_name = policy_name
                    results.append(result)
                    logger.info(
                        f"  → reward={result.reward:.3f}, "
                        f"steps={result.steps}, "
                        f"time={result.duration_seconds:.1f}s"
                    )
                except KeyboardInterrupt:
                    logger.warning("  → Interrupted, saving partial results...")
                    raise
                except Exception as e:
                    logger.error(f"  → FAILED: {e}")
                    results.append(EvalResult(
                        question_id=q["id"],
                        question=q["question"],
                        policy_name=policy_name,
                        reward=0.0,
                        answer_score=0.0,
                        citation_precision=0.0,
                        citation_recall=0.0,
                        efficiency_bonus=0.0,
                        steps=0,
                        duration_seconds=0.0,
                    ))
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
    except KeyboardInterrupt:
        logger.warning("Eval interrupted — returning partial results")

    return results
