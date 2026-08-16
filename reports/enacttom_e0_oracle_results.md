# EnactToM E0: Oracle State and Reducer

## Scope

- Task entries: 300
- Unique task IDs: 289
- First-order probe cases: 317
- Higher-order probe cases: 303 (diagnostic only)

## Deterministic Replay

| Metric | Value |
| --- | ---: |
| Physical goals | 1013 |
| Physical goals satisfied | 986 |
| Physical goal replay rate | 97.33% |
| Tasks with all goals satisfied | 291 |
| Tasks with failed goal replay | 9 |
| Invalid golden actions | 0 |
| Golden access violations | 83 |
| Tasks with access violations | 76 |
| Failed tasks with no effectful golden action | 9 |

Action coverage:

| Action | Count |
| --- | ---: |
| Navigate | 2782 |
| Open | 293 |
| Pick | 649 |
| Place | 649 |
| Wait | 1526 |

## First-order Message Interventions

| Metric | Value |
| --- | ---: |
| Evaluated message cases | 152 |
| Oracle delivery rate | 100.00% |
| Delete target sensitivity | 100.00% |
| Delete non-target invariance | 100.00% |
| Redirect cases | 71 |
| Redirect target sensitivity | 100.00% |
| Redirect non-target invariance | 100.00% |

Case status:

| Status | Count |
| --- | ---: |
| evaluated | 152 |
| skip_no_grounded_route | 22 |
| skip_target_already_observed | 143 |

## Public-data Diagnostics

The reducer found no unsupported action syntax. It did find two dataset-level consistency signals:

- 76 tasks contain 83 golden navigation steps whose resolved rooms appear in the acting agent's room restrictions. This is a static metadata/action mismatch and requires Habitat runtime confirmation before being called a benchmark error.
- 9 task entries have unsatisfied saved planner literals; all of them contain only Navigate/Wait actions and no effectful golden action.

| Split | Failed replay task ID |
| --- | --- |
| standard | `emtom-102815859-1708-cooperative-staging-checklist-tray-sorting-and-fridg-78db5cf8` |
| standard | `emtom-104348082_171512994-1776-cooperative-quiet-setup-move-kettle-stage-cup-open-c-e60d402a` |
| standard | `emtom-102344049-1635-mixed-gallery-lighting-reset-with-remote-confi-0d7ac6b4` |
| hard | `emtom-107734479_176000442-1919-cooperative-open-the-hidden-side-table-for-the-readi-d3db7c82` |
| hard | `emtom-107734479_176000442-740-mixed-guest-reset-with-a-kitchen-pickup-and-la-016e0b77` |
| hard | `emtom-105515448_173104512-677-cooperative-open-the-hidden-tv-room-table-for-the-in-f6e792a9` |
| hard | `emtom-108736872_177263607-1613-cooperative-guest-ready-relay-with-a-hidden-living-r-ec5fd947` |
| hard | `emtom-107734449_176000403-1119-mixed-choose-the-correct-airplane-for-the-offi-e7ee9457` |
| hard | `emtom-104862669_172226853-1827-mixed-choose-the-right-kitchen-item-for-the-di-8af9f8ff` |

## Claim Boundary

This validates query-independent state compilation, physical replay, and first-order intervention locality. Oracle messages are synthetic probe-grounded interventions, not recovered benchmark communication or learned-model evidence.

The public golden trajectories contain no communication actions. E0 therefore tests reducer and intervention semantics, not whether an agent can infer, generate, or execute the required messages.
