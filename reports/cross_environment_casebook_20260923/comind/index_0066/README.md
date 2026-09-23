# Same SAM, different identity: dough localization

**Environment:** CoMind  
**Scope:** Flash symmetric SAM300 development set; static visual grounding; no new VLM/SAM calls

## 1  Frozen inputs

Task: joint_attention.

Transcript: should

## 2  Same SAM rule

Direct identity: food drink or container.

SSC identity: dough.

Same final-field selection rule and SAM backend.

## 3  Localization result

Direct+SAM IoU: left_bbox=0.003, right_bbox=0.000

SSC+SAM IoU: left_bbox=0.925, right_bbox=0.873

## Limits / interpretation

- Same candidate-selection rule does not imply the same generated identity or chosen view.
- Case66 social_code is unknown; it is not evidence for a social-layer contribution.
- Case226 selects different views (Direct helper, SSC leader); do not attribute the difference solely to noun specificity.
- Gold overlays are evaluation-only. Raw SAM boxes are xyxy in1408-square views; scores use converted xywh on the combined canvas.
- Frozen scorer uses IoU>0.5. No selected case lies exactly at0.5. All non-localization field scores are preserved.

## Files

- `case.json`: complete selected saved records, labels and provenance.
- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.
- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.
- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.
