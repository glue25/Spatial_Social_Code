# Executable SSC-generated Habitat cases

All cases use the same HSSD stage, episode, spawns, physical initial state, goals and seed. 
Only the owner of one private location fact changes; the fourth case adds an irrelevant fact.

| Case | Key fact owner | Direct | Dynamic SSC |
|---|---|---:|---:|
| fact_owner_agent0 | agent_0 | 2/3 | 3/3 |
| fact_owner_agent1 | agent_1 | 3/3 | 3/3 |
| fact_withheld | withheld | 2/3 | 2/3 |
| irrelevant_fact_control | agent_1 | 3/3 | 2/3 |

These are mechanism cases, not an accuracy benchmark. Golden trajectories, target answers, and persisted literal-ToM expected responses were removed before execution.
