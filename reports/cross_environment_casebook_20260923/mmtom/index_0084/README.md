# Container inspection and a recorded belief change

**Environment:** MMToM  
**Scope:** Flash historical600; static frame+text compilation; no dense refinement in selected cases

## 1  Observed task sequence

Robert opens and closes the third kitchen cabinet while looking for a dish bowl.

## 2  Actual intermediate code

Generated records include an earlier inside belief and a later not-inside belief.

The final answer also uses deterministic inspection evidence and an option-conditioned posterior.

## 3  Saved prediction

Direct: A (wrong).

SSC: B (correct).

This is a historical pipeline example, not an option-blind update experiment.

## Limits / interpretation

- Static video compilation, not an online state-store before/after trace.
- Generated belief records are hypotheses, not direct proof of attention or belief.
- The full historical code includes option-conditioned query evidence, rule bridges and posteriors.
- Case0 was excluded: raw Direct ends in B but archived parsed prediction is A; original aggregate scores were not modified.

## Files

- `case.json`: complete selected saved records, labels and provenance.
- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.
- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.
- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.
