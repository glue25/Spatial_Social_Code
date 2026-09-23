# Goal inference with a historical answer-conditioned scorer

**Environment:** MMToM  
**Scope:** Flash historical600; static frame+text compilation; no dense refinement in selected cases

## 1  Observed task sequence

Jessica inspects kitchen appliances; the question supplies a belief premise and asks which object she is seeking.

## 2  Actual intermediate code

The neural social code favors salmon.

Historical query evidence and Bayesian fusion change the final answer toward apple.

## 3  Saved prediction

Direct: A (wrong).

SSC final: B (correct).

The successful endpoint cannot be attributed to the neural code alone.

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
