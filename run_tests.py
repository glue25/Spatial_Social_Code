"""Run source-only tests without model requests or original result archives."""
from pathlib import Path
import os, subprocess, sys
ROOT = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
env["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in (".", "benchmarks/ma_egoqa", "spatial_social_code"))
subprocess.run([sys.executable, "-m", "unittest", "-q", "test_spatial_social"], cwd=ROOT/"spatial_social_code", env=env, check=True)
tests = ["tests/test_information_history_v1.py", "tests/test_comind.py", "tests/test_comind_optimization.py", "tests/test_enacttom.py", "experiments/habitat_closed_loop/test_closed_loop.py", "benchmarks/ma_egoqa/egomas/src/test_jdcloud_baseline.py", "benchmarks/ma_egoqa/egomas/src/test_spatial_social_jdcloud.py", "benchmarks/ma_egoqa/egomas/src/test_video_spatial_social_jdcloud.py", "benchmarks/ma_egoqa/egomas/src/test_video_grounded_v2_jdcloud.py", "benchmarks/ma_egoqa/egomas/src/test_video_provenance_v3_jdcloud.py"]
subprocess.run([sys.executable,"-m","pytest","-q","-p","no:cacheprovider",*tests], cwd=ROOT, env=env, check=True)

# Test relocated snapshots in isolated processes with their own import roots.
for folder, targets in [("benchmarks/comind/pipelines/sam", ["test_sidecar.py"]), ("benchmarks/information_history", ["tests/test_information_history_v1.py"]), ("benchmarks/habitat", ["tests/test_qdd_habitat.py"])]:
    snapshot_env = dict(env, PYTHONPATH=str(ROOT / folder))
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets], cwd=ROOT/folder, env=snapshot_env, check=True)
