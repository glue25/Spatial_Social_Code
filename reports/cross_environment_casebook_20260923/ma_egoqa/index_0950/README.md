# Self-report linked to a belief record and cited answer

**Environment:** MA-EgoQA  
**Scope:** Flash rich100; same compiled record across three arms; static audiovisual QA

## 1  Actual question / evidence

Why did Alice feel the smell of food was making her unwell?

## 2  Compiled links

S2/S3 record Alice saying the smell makes her feel full. The belief record and final answer refer to those same evidence IDs.

## 3  Saved answers

ssc_no_attribution: D

ssc: D

ssc_no_references: D

Gold: D.

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
