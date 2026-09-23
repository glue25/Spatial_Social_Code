# formal_019: tray_5

Counterexample: SSC makes state errors while prose answers all four histories correctly.

**Constructed symbolic history; not a physical Habitat rollout or a video.**
All four histories and all four arms are included. Editorial/post-hoc selection; not aggregate evidence.

World: `fridge_49` → `table_28`. Target: `agent_2`; unrelated: `agent_0`.

| History | Gold target | SSC target | Prose target | No links | No attribution |
|---|---|---|---|---|---|
| witness | table_28 | fridge_49 | table_28 | fridge_49 | unknown |
| unseen | fridge_49 | fridge_49 | fridge_49 | table_28 | unknown |
| delivered | table_28 | table_28 | table_28 | fridge_49 | unknown |
| undelivered | fridge_49 | fridge_49 | fridge_49 | table_28 | unknown |

`case.json` includes full physical/social records, joins, StateStore snapshots, oracle access logs and scores.
`saved_calls/` contains every original request/response for the selected base.
See `history_panel.png` (also PDF/SVG); no intermediate trend was inferred from endpoint scores.

The ablated input can be underdetermined: `unknown` may be rational even when it fails full-information gold.
The favorable and reverse contrasts do not establish overall superiority of either serialization.
