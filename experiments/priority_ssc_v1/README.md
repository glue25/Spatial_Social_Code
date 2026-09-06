# Priority SSC experiments

This package implements the first three requested experiments without rerunning
the completed component/SAM ablations.

## 1. Same-input, same-model reasoning baselines

`run_reasoning_baselines.py` compares CoT, a natural-language scratchpad, a
PDDL-Mind-style adaptation, Spatial-only, and SSC.  Every condition receives
the identical frozen Physical trace and makes exactly one deterministic final
reasoner call.  SSC additionally receives the frozen agent-indexed Social
state.  Posterior, candidate-answer, query-evidence, and action-evidence
records are removed from that state.  The PDDL condition is an adaptation on
the common trace, not an official PDDL-Mind reproduction.

The runner checkpoints every example and reports accuracy, actual calls,
input/output tokens, retries, latency, and paired McNemar tests.

```bash
python experiments/priority_ssc_v1/run_reasoning_baselines.py \
  --dataset mmtom --backend local-qwen32b --total 100
```

## 2. Two-reviewer intermediate-state audit

`prepare_human_audit.py` deterministically samples 100 unique MMToM episodes
and 100 MA-EgoQA questions with no reused exact source clip. It emits a
target-answer-blind packet plus separate reviewer A/B CSV files. Reviewers use
`yes`, `no`, or `uncertain` for state correctness and citation support. For a
wrong state they must enter corrected trace-record JSON or `DROP`.

Reviewers must work independently. Run the aggregator only after both files
are complete:

```bash
python experiments/priority_ssc_v1/aggregate_human_audit.py
```

It reports raw agreement and Cohen's kappa, separately reports Physical-state
and Social/access accuracy, and asks for adjudication whenever a claim is not
unanimously correct and supported. The resulting `verified_states.jsonl`
contains no question, option, gold label, model answer, or correctness field.

## 3. Four-cell state correction

`run_four_cell.py` consumes only the completed verified states and compares:

- predicted Physical + predicted Social;
- human Physical + predicted Social;
- predicted Physical + human Social;
- human Physical + human Social.

All four cells use the byte-identical final prompt template and one call to the
same frozen reasoner. The primary contrast is human/predicted vs human/human.
The report also isolates cases where the predicted Physical state was already
confirmed correct and tests predicted/predicted vs predicted/human.

## Unattended queue

`run_priority_background.sh` runs experiment 1, waits at the genuine human
annotation gate, runs experiment 3 after adjudication, exports aggregate-only
reports, and calls `publish_results.sh`. Raw model outputs, video paths, and
human annotation sheets remain ignored by Git.

## 4–5. Attribution intervention and second backbone

`run_attribution_intervention.py` builds 100 balanced minimal pairs. Within a
pair the Physical terminal state is byte-identical; only the observer, message
recipient, or presence fact changes. Every pair also changes an unrelated
person/event as a negative control. The report includes pair-both-correct,
target-update, and negative-control invariance.

`run_cross_model_background.sh` repeats the five core reasoning baselines and
the attribution intervention with the configured Gemini endpoint. Together
with the local Qwen3-VL-32B run, this provides the two-backbone comparison.
Both runners save actual input/output token counts, calls, retries, and
latency; missing service-side token fields remain zero/missing rather than
being estimated.
