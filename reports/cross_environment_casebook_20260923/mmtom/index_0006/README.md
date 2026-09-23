# Counterexample: recorded belief versus fused final answer

**Environment:** MMToM  
**Scope:** Flash historical600; static frame+text compilation; no dense refinement in selected cases

## 1  Observed task sequence

Elizabeth searches kitchen containers and then approaches a living-room cabinet while seeking a wine glass.

## 2  Actual intermediate code

Generated social code records an inside-cabinet belief.

The fused historical answer nevertheless selects the opposite option.

## 3  Saved prediction

Direct: A (correct).

SSC final: B (wrong).

Useful for showing that intermediate records do not guarantee a correct final answer.

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
