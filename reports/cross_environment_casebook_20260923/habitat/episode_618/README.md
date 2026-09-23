# Bottle location report -> targeted navigation

**Environment:** Habitat  
**Scope:** Online Dynamic SSC; Flash; seed891011; episode618; job42

## 1  Before delivery

agent_0: no agent_2-attributed report for chest_of_drawers_33 in the pre-update state.

## 2  New attributed report

Turn 1: agent_2 -> agent_0.

bottle_2 is on chest_of_drawers_33 in bedroom_2. I'm barred from there; please get it and put it on counter_25.

## 3  Next actual action

Turn 2: Navigate[chest_of_drawers_33].

Navigation feedback: succeeded.

Final shared goals: 3/3 (not the immediate navigation result).

## Limits / interpretation

- New reported claim, not verified object-location truth; do not label as a corrected false belief.
- Actual next request, delta and action verified; planner causality is not isolated.
- Bottle placement later requires inter-agent transfer. Do not compress it into this next navigation.

## Files

- `case.json`: complete selected saved records, labels and provenance.
- `spatial_code.py` / `social_code.py` where present: literal generated code, not automatically executed.
- `input_preview.jpg` where present: labeled visual preview; see media manifest for exact scope.
- `evidence_flow.png/pdf/svg`: compact editorial summary, not new model output.
