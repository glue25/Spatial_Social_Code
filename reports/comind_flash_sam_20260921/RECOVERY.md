# Four-case independent recovery

All4 original SSC truncations selected before calls. Direct and SSC freshly regenerated at uniform32768, then SAM. No primary300 prediction or score was replaced. This is a failure-selected, higher-budget diagnostic, not a corrected primary total.

| Index | Task | Direct valid | SSC valid | SAM state | Direct fields correct | SSC+SAM fields correct |
|---|---|---|---|---|---:|---:|
| 94 | joint_attention | True | True | complete | 1/4 | 1/4 |
| 236 | handover | True | True | complete | 1/6 | 1/6 |
| 113 | scoia | True | True | complete | 3/4 | 3/4 |
| 245 | handover | True | True | complete | 1/6 | 1/6 |

Per-field counts are descriptive and task field counts differ. Validity does not imply correct localization or semantics. Original300 result remains Direct17.21%, SSC+SAM22.21% (+5.00pp macro14).
