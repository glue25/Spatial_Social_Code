# A guess, a correction, and explicit evidence references

**Environment:** MA-EgoQA  
**Scope:** Flash rich100; same compiled record across three arms; static audiovisual QA

## 1  Actual question / evidence

Who assumed Katrina had a boyfriend when she simply sat down, but turned out to be wrong?

## 2  Compiled links

S2 records Tasha asking about a boyfriend; S3 records Jake replying no. Social records retain the attributed guess and correction.

## 3  Saved answers

ssc: C

ssc_no_references: C

ssc_no_attribution: C

Gold: C.

## Limits / interpretation

- All three arms see the same clips and derive views from one compiled record; the final answer still sees audiovisual clips.
- The input row.answer is an intentional option0 dummy, not ground truth. This export uses SCORER_ONLY_gold.json.
- Quotes, speaker identities and inferred beliefs have not received independent full audio/identity annotation review.
- Full100 outcomes:46/46/49 correct for full/no-references/no-attribution. Individual examples do not establish linkage benefit.
- Case1597 no-references is invalid, not a valid wrong answer; do not claim an evidence-link mechanism success from it.

## Files

- `case.json`: complete selected saved records, labels and provenance.
- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.
- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.
- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.
