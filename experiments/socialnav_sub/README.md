# SocialNav-SUB integration

This adapter treats every navigation scene as the reusable unit.  A scene is
compiled once, before any question or answer choice is supplied.  All questions
for that scene then share the same frozen Spatial and Social Code.

The canonical input schema is intentionally small:

```json
{
  "scene_id": "scene_name",
  "goal": {"x": 5.0, "y": 0.0},
  "samples": [
    {
      "t": 0,
      "robot": {"x": 0.0, "y": 0.0, "yaw": 0.0},
      "people": {"person_1": {"x": 2.0, "y": 0.2}}
    },
    {
      "t": 4,
      "robot": {"x": 1.0, "y": 0.0, "yaw": 0.0},
      "people": {"person_1": {"x": 2.2, "y": 0.1}}
    }
  ]
}
```

Compile with:

```bash
python spatial_social_code/socialnav_sub.py \
  --input workdirs/socialnav_sub/pilot/scenes.jsonl \
  --output workdirs/socialnav_sub/pilot/compiled.jsonl
```

The first pilot freezes six scenes and reports Direct, Spatial-only,
Social-only, and Spatial+Social under the same visual input.  Human labels are
scored with the official probability-of-agreement metric.  Ground-truth
trajectory/BEV annotations are reported separately as an oracle condition and
are not mixed into the visual-input headline result.

The frozen JD Cloud pilot uses one question from each official reasoning group
per scene (18 questions total).  It compares direct visual answering, oracle
code, and visually predicted Spatial+Social representation.  The visual
compiler is called once per scene before any question is constructed, and all
three questions reuse that frozen output.
