# Habitat closed-loop three-seed extension

Seeds: 891011, 891012, 891013; 12 tasks; 108 real rollouts.

This is a multi-seed extension of the engineering configuration, not a frozen final protocol.

| Condition | Strict success | Mean completion | Benchmark success | API calls | Tokens |
|---|---:|---:|---:|---:|---:|
| direct | 3/36 (8.3%) | 53.0% | 8.3% | 56.2 | 147432 |
| static_ssc | 3/36 (8.3%) | 47.2% | 8.3% | 56.9 | 168007 |
| dynamic_ssc | 4/36 (11.1%) | 41.7% | 13.9% | 57.5 | 221910 |

## Paired comparisons

- static_ssc_vs_direct: candidate-only=2, direct-only=2, exact McNemar p=1.
- dynamic_ssc_vs_direct: candidate-only=4, direct-only=3, exact McNemar p=1.

Task-seed rows are paired, but repeated seeds of the same task are not claimed as independent task clusters.
