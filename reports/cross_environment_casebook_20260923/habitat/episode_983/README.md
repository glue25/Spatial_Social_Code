# Unsuccessful search -> attributed jug report -> navigation

**Environment:** Habitat  
**Scope:** Online Dynamic SSC; Flash; seed891011; episode983; job37

## 1  Before delivery

agent_2: no agent_0-attributed report for bed_9 in the pre-update state.

## 2  New attributed report

Turn 5: agent_0 -> agent_2.

agent_2, please get jug_3 from bed_9 and put it on table_13. I'm getting book_0.

## 3  Next actual action

Turn 6: Navigate[bed_9].

Navigation feedback: succeeded.

Final shared goals: 2/3 (not the immediate navigation result).

## Limits / interpretation

- New reported claim, not verified object-location truth; do not label as a corrected false belief.
- Actual next request, delta and action verified; planner causality is not isolated.
- Episode983 ends at2/3; its NL pair had a service problem. Do not present as a clean superiority comparison.

## Files

- `case.json`: complete selected saved records, labels and provenance.
- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.
- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.
- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.
