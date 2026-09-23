# Cross-environment casebook (13 audited examples)

See **[the complete Chinese guide and case index](README_zh.md)**.

- [Habitat](habitat/): 3 online state-update traces, including the actual next
  request/action and saved frame. Primary example: [inspection588](habitat/inspection_588/).
- [MA-EgoQA](ma_egoqa/): 4 static audiovisual evidence-link examples, including
  a concurrency error and an invalid-output diagnostic. Audio/speaker semantics
  still require human sign-off; these are not online memory traces.
- [MMToM](mmtom/): 3 explicitly historical, answer-conditioned code examples,
  including a negative case. Not option-blind mechanism evidence.
- [CoMind](comind/): 3 same-SAM grounding comparisons, with all field scores,
  native-frame overlays, and a negative case. Not blanket social-layer evidence.

All cases are selected post hoc. No new experiments were run. Original outputs,
hashes, full selected code and uncertainty/limitations accompany each example.
Never pool these examples into a new success-rate estimate.

Offline verification: `python -m experiments.verify_cross_environment_casebook`.
Re-render exported figures: `python -m experiments.build_cross_environment_casebook --from-export`.
The main guide explains dependencies, provenance and media scope.
