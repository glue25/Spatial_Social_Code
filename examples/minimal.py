"""Execute a hand-written linked scene: python -m examples.minimal.

The scene and belief below are illustrative input records, not model predictions.
No dataset, model, network request or file output is used.
"""
import json
from spatial_social_code.runtime import execute_social, execute_spatial

SPATIAL_CODE = """
S.fact(0, "inside", "cup", "cabinet", source="example", evidence_id="S1")
S.visible(0, "Alice", "cup", True, source="example", evidence_id="S2")
"""
SOCIAL_CODE = """
M.belief(0, "Alice", "The cup is inside the cabinet.", 1.0,
         "Hand-written example premise", evidence_ids="S1,S2")
"""


def main():
    spatial = [json.loads(line) for line in execute_spatial(SPATIAL_CODE).splitlines()]
    ids = {row["values"]["evidence_id"] for row in spatial}
    social = [json.loads(line) for line in execute_social(
        SOCIAL_CODE, valid_evidence_ids=ids, require_evidence=True
    ).splitlines()]
    print(json.dumps({"spatial": spatial, "social": social}, indent=2))


if __name__ == "__main__":
    main()
