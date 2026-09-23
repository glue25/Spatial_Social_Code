# Information-history diagnostic v1 — frozen before inference

## What this experiment is, and is not

This is a **constructed symbolic, fixed-record reading diagnostic**, not a new
Habitat physics run or visual counterfactual. Existing EnactToM task files provide
entity vocabulary and episode keys. Their original tasks, private secrets, golden
trajectories and camera visibility are NOT replayed. Consequently the study can
support only controlled **text-state semantics and information-use** claims.

The existing `habitat_closed_loop.state_store.StateStore` stores the individually
acquired experiences. A new factorized diagnostic adapter exports physical events,
social access records and explicit joins. This is **not** claimed to be the byte-
identical historical Habitat planner input. Source hashes identify this exact new
adapter and scorer; no current commit is used to certify historical runs.

## Predefined minimal scope

* Exclude already materialized task episode keys under four listed historical
  roots, plus episode742 inspected during design. Record every exclusion path/hash.
* From the released standard/hard task corpus, require >=3 agents, >=2 objects,
  >=3 locations; deduplicate `(scene_id, episode_id)` by fixed hash rank.
* First10 eligible ranked episodes are engineering only. Next40 are formal.
  No success labels or predictions are used. This is local-provenance independence,
  NOT a guarantee that task vocabulary was globally unseen or absent from training.
* Each base: witnessed move; unseen move without relevant report; unseen move
  with delivered relevant report; attempted relevant report not delivered.
* Physical events, final positions and unrelated-agent state are invariant within
  each base. Role assignments, location assignments and presentation order are
  deterministic and frozen; half have an unobserved intermediate physical move.
* Four arms: linked SSC diagnostic, deterministic lossless prose, no links,
  no attribution. 10x4x4=160 engineering and 40x4x4=640 formal final calls.
  At most four initial key probes (up to three transport attempts each).

The engineering gate requires >=95% valid outputs and all structural/unit tests.
**No accuracy threshold** is used. Passing automatically starts all640 calls;
failure pauses for engineering review. Engineering outcomes never change the
formal sample selection. All formal errors, invalid outputs, exhausted transport
failures and timeouts remain in the intended denominators; coverage distinguishes
pending calls from terminal failures.

Engineering scoring revision2 separates syntax from semantics: a four-string-field
JSON response remains format-valid when a location or evidence ID is wrong. That
field scores incorrect; correct answers in other fields retain credit. Malformed
JSON, missing/wrong-typed fields and truncated responses still fail the schema.
The initial engineering snapshot incorrectly rejected out-of-enum evidence IDs as
whole-response invalid. It is retained, paused, with all outputs. The revised run
reuses exactly the same cached responses, prompts, oracle, IDs and generation
settings, recomputes scores without model calls, and starts formal inference only
under revision2. No formal response was used to choose this correction.

## Factor definitions and limitations

Physical events contain no holder/recipient name. Access records contain no
duplicated object/location description; only explicit links identify which event
was accessed. Simultaneous target/distractor updates and report contents prevent
timestamps alone from recovering the mapping. Reports are truthful and trusted by
declared construction; confirmed vision means acquisition, NOT mere visibility.

* **prose** preserves every field/value/list position, checked by deterministic
  round trip. It is a serialization control, not a learned summary or independent
  dynamic natural-language updater. Generation budgets are identical; input token
  counts are measured, not asserted equal.
* **no_links** deletes the join table only; every physical/social record, identity,
  status and time remains byte-equivalent. WITNESS/UNSEEN/DELIVERED then become
  identical prompts for each base despite different target gold labels.
  This explicitly demonstrates an information-loss ceiling! A drop is evidence
  that the removed mapping was needed, NOT superiority at equal information.
* **no_attribution** replaces holder and source-agent values with null, retaining
  records and evidence links. It does not swap names or introduce false facts.
  No grouping, holder-derived readable IDs or positional role mapping survives.
  Query agent names remain in the question, never in the ablated records.
* The intended receiver does NOT get the failed packet in its StateStore; a
  separate unavailable-attempt audit record is visible to the external evaluator.
  This evaluator is not an acting agent given extra private information.

The oracle separately replays world events and actual acquired observations /
delivered messages. It verifies location and exact latest supporting evidence,
not just whether a citation ID exists. Initial beliefs persist without updates.
Unknown is a valid output and may be the rational ablated answer, but it does not
match the full-information private-state gold. Report this asymmetry prominently.

## Metrics / statistics fixed in advance

Primary outcome: average paired joint correctness of world, target and unrelated
state across four changed-information contrasts, evaluated per base episode.
Co-primary comparisons: SSC minus no_links and SSC minus no_attribution.
10,000 paired base-cluster bootstrap resamples, fixed seed20260923; Bonferroni
97.5% marginal intervals for these two comparisons. SSC vs prose is secondary95%.
Also report world invariance **and** correct invariance, target correct change,
unrelated invariance **and** correct invariance, field accuracy, evidence-supported
target accuracy and all-four-history joint correctness. All40 base episodes stay
in all arm denominators.40 episodes is NOT a power guarantee; shared templates
limit generalization even when base-cluster intervals are narrow.

## Model / cost / unattended operation

Same provisioned `gemini-3-flash-preview`, temperature0, reasoning_effort=low,
max_tokens=2048, JSON output contract for every arm. Exact provider model string
must match; this is provider reporting, not independently verified model weights.
Per-key shared locks prevent concurrent requests from this pipeline and compatible
legacy workers. Available keys are determined by technical preflight, not scores.
Conditions/histories interleave in a frozen randomized order within every base.
Max12 attempts for transport errors; no resampling of any HTTP200 model output,
including invalid JSON, truncation or wrong answers. Save every attempt and actual
token usage; missing usage remains missing. Retry charges may be unobservable after
network timeouts. Key failover cannot change model, prompt or generation parameters.

Preparation creates immutable `dataset.json`, `selection_audit.json`,
`frozen_config.json`, copied source and all50 source task definitions. Run only
from the snapshot. API keys remain in existing credential files, never copied.

```bash
PYTHONDONTWRITEBYTECODE=1 MA-EgoQA/.venv/bin/python -m \
  experiments.information_history_v1.run prepare --output /absolute/new/output
# Then cd /absolute/new/output/source, use the same absolute Python executable:
python -m experiments.information_history_v1.run run --output /absolute/new/output
```

`status.json`, per-base `workers/`, `calls/`, engineering gate, condition costs and
paired summaries are saved automatically. Create an output-root `PAUSE` file to
stop at the next request boundary. Resume with the same command; completed request
hashes must match. `summarize` exports partial results with explicit pending counts.
No second-backbone, extra seed, large QA expansion or new case-mining is scheduled.
