# CoMind optimization v2

This branch performs train-only, recording-level out-of-fold diagnosis and
offline routing. It never reads hidden CoMind test labels and never performs a
benchmark submission.

Run:

```bash
python experiments/comind_optimization_v2/comind_v2.py
python experiments/comind_optimization_v2/test_comind_v2.py
```

The optimizer evaluates explicit `xywh` and `xyxy` hypotheses, preserves
identity candidates extracted from Spatial traces, and calibrates all source
selection/canonicalization on the other recording folds. Outputs are written
only to `workdirs/comind_optimization_v2/`.

The GPU experiment is frozen, but was not run because `nvidia-smi -L` reports
no available NVIDIA device:

```bash
python experiments/comind_optimization_v2/prepare_gpu_manifest.py
python experiments/comind_optimization_v2/run_gpu_grounding.py \
  --model-path /pfs/xiabin/models/Qwen3-VL-8B-Instruct
python experiments/comind_optimization_v2/evaluate_gpu_grounding.py \
  workdirs/comind_optimization_v2/gpu_grounding_predictions.jsonl
```

The identity-first two-stage pilot avoids asking the VLM to invent numeric
boxes. Start the existing SAM3.1 service on the GPU, then run ten frozen rows:

```bash
cap-x/.venv-libero/bin/python cap-x/capx/serving/launch_sam3_server.py \
  --checkpoint-path /pfs/zhengshenghe/code/WAM/WAM-DreamZero/data_processing/seg/model/sam3.1/sam3.1_multiplex.pt \
  --port 8114

cap-x/.venv-libero/bin/python \
  experiments/comind_optimization_v2/run_two_stage_grounding.py --limit 10

python experiments/comind_optimization_v2/evaluate_gpu_grounding.py \
  workdirs/comind_optimization_v2/two_stage_grounding_predictions.jsonl \
  --output workdirs/comind_optimization_v2/two_stage_grounding_evaluation.json
```

The two-stage runner stores SAM proposals from every target view, writes after
each sample, and resumes completed sample IDs. It uses a separate output path,
so it never overwrites the stopped one-stage pilot.

For a balanced 30-row smoke test (10 per task), use `--per-task 10`. Joint
attention and handover annotations use the official half-scale horizontal
mosaic coordinates (leader on the left, helper on the right); SCOIA remains in
the individual 1408x1408 helper-frame coordinates. The scorer performs these
task-specific transforms explicitly.
