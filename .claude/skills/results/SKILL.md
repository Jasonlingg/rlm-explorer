---
name: results
description: Summarize the latest eval run into a markdown results file
allowed-tools: Bash, Read, Write, Glob
---

# Summarize Eval Results

Find the most recent eval transcript in `out/` and generate a markdown summary at `out/hard_eval_results.md`.

## Steps

1. Find the latest `out/run_*.json` file by modification time
2. Read it and compute:
   - Per-policy average reward, answer accuracy, citation precision, citation recall, avg steps
   - Per-question breakdown across all policies
   - Failure analysis for claude_policy: identify timeouts (max steps + 0 reward), early quits (1 step + low reward), errors (0 steps)
   - Which questions claude_policy won vs lost against baselines
3. Write the summary to `out/hard_eval_results.md` using this format:

```markdown
# Hard Eval Results — {date}

Run: `{filename}`
Command: `{reconstruct from context}`
Agent model: {from transcript or default claude-haiku-4-5-20251001}

## Summary

| Policy | Avg Reward | Avg Answer | Avg Steps |
|--------|-----------|-----------|-----------|
| ... sorted by reward descending ... |

## Per-Question Breakdown

| Question | claude_policy | naive_rag | context_stuffing | single_shot |
|----------|--------------|-----------|-----------------|-------------|
| h01 | reward (NOTES) | reward | reward | reward |
| ... |

## Failure Analysis — claude_policy

Table of failed questions with step count and diagnosed issue.
Read the trajectory to determine the failure mode:
- TIMEOUT: hit max steps without submitting
- EARLY QUIT: submitted on step 1 without exploring
- ERROR: 0 steps, crashed
- SYNTAX LOOP: repeated SyntaxErrors
- PAGINATION LOOP: burned steps on read() slicing instead of search_within()

## Successful Questions — claude_policy

Table of questions where claude_policy scored > 0.5 with what worked.

## Key Takeaways

Brief analysis of what's working and what isn't.
```

4. Print the path to the generated file when done.

If `$ARGUMENTS` contains a path to a specific run file, use that instead of the latest.
