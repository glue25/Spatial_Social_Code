"""Build an evidence-backed Pro MMToM replacement panel, without API calls.

Only writes a new asset directory; never edits the other Figure2 panels.
Original composite/manuscript source is not present in this workspace.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

PACK = ROOT / "reports/flash_full_results_20260920"
OUT = ROOT / "reports/figure2_pro_replacement_20260923"


def final_answer(raw):
    """Response-only explicit final-line reader; no gold-dependent extraction."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None, "unresolved"
    line = lines[-1]
    if line.startswith("**") and line.endswith("**"):
        line = line[2:-2].strip()
    for label, prefix in [("bare_final_line", ""), ("answer_final_line", r"Answer\s*[:=]\s*"),
                          ("explicit_statement_final_line", r"The correct statement is\s+")]:
        match = re.fullmatch(prefix + r"(?:([AB])|\*\*([AB])\*\*)[.]?", line, re.I)
        if match:
            return next(x for x in match.groups() if x).lower(), label
    return None, "unresolved"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(PACK / "mmtom/pro_results.jsonl.gz", "rt") as f:
        records = [json.loads(line) for line in f]
    inputs = {r["index"]: r for r in json.loads((PACK / "mmtom/original_frozen_inputs/inputs.json").read_text())}
    ledger = []
    for record in records:
        final, policy = final_answer(record["direct_raw"])
        candidate = final is not None and final == record["direct_prediction"] and final != record["answer"] and record["correct"]
        ledger.append({"index": record["index"], "question_type": record["question_type"], "explicit_direct": final,
                       "stored_direct": record["direct_prediction"], "gold": record["answer"],
                       "ssc_prediction": record["prediction"], "candidate": bool(candidate),
                       "dense": record["uncertainty_triggered"]})
    eligible = [r for r in ledger if r["candidate"] and not r["dense"]]
    index = min(r["index"] for r in eligible)
    assert index == 302
    record = next(r for r in records if r["index"] == index)
    item = inputs[index]
    assert record["direct_raw"] == "B" and record["answer"] == item["row"]["answer"] == "a"
    assert record["prediction"] == "a" and not record["uncertainty_triggered"]
    assert record["image_details"] == item["frames_8"]
    assert final_answer(records[0]["direct_raw"])[0] is None
    # Re-execute only literal archived SSC calls in the frozen sandboxed runtime.
    legacy = PACK / "code/workdirs/flash_model_only_legacy_20260918"
    sys.path.insert(0, str(legacy))
    import legacy_loader
    legacy_loader.load()
    from runtime import execute_spatial, execute_social
    from social_code import answer_posterior_probs
    with patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden")):
        assert execute_spatial(record["spatial_code"]) == record["spatial_trace"]
        replayed_social = execute_social(record["social_code"])
        saved_rows = [json.loads(line) for line in record["social_trace"].splitlines()]
        replay_rows = [json.loads(line) for line in replayed_social.splitlines()]
        schema_additions = []
        assert len(saved_rows) == len(replay_rows)
        for position, (saved, replayed) in enumerate(zip(saved_rows, replay_rows)):
            for key, default in (("constraint", "ALL"), ("variable_type", "unknown")):
                if saved["kind"] == "query_candidate" and key not in saved["values"]:
                    assert replayed["values"].pop(key) == default
                    schema_additions.append({"row": position, "field": key, "added_default": default})
            assert replayed == saved
    posterior = answer_posterior_probs(record["social_trace"])
    assert posterior == {"A": record["prob_a"], "B": record["prob_b"]}
    social_lines = record["social_code"].splitlines()
    selections = {
        "physical_fact": record["spatial_code"].splitlines()[0],
        "attributed_belief": next(line for line in social_lines if "given query premise" in line),
        "query_rule": [line for line in social_lines if "visited location is believed not to contain" in line],
        "posterior": [line for line in social_lines if "M.posterior(-1" in line and "answer_goal" in line],
    }
    hashes = json.loads((PACK / "mmtom/original_frozen_inputs/media_hashes.json").read_text())
    frame_records = []
    for position, number in enumerate(item["frames_8"]["frame_numbers"]):
        saved = next(p for p in item["media_files"] if p.endswith(f"/Action_{number:04d}_0_normal.png"))
        path = Path(saved.replace("${PROJECT_ROOT}", str(ROOT)))
        assert sha(path) == hashes[saved]
        dest = OUT / "mmtom_pro_0302/frames" / f"input_{position+1:02d}_frame_{number:04d}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        frame_records.append({"position": position+1, "frame_index": number, "file": str(dest.relative_to(OUT)),
                              "source": saved, "sha256": sha(dest), "time_seconds": None})
    folder = OUT / "mmtom_pro_0302"
    write(folder / "runtime_schema_audit.json", {"social_byte_exact": replayed_social == record["social_trace"],
          "all_original_fields_equal": True, "known_default_field_additions": schema_additions,
          "caveat": "Available Flash-frozen runtime adds two defaults to each query_candidate relative to older Pro trace. This is not per-run Pro source-commit verification."})
    for name in ("spatial_code", "social_code"):
        (folder / f"{name}.py").write_text(record[name]+"\n")
    (folder / "direct_raw.txt").write_text(record["direct_raw"]+"\n")
    write(folder / "case.json", {
        "index": index, "episode": item["row"]["episode"], "model_label": record["model"],
        "frozen_benchmark_row": item["row"], "saved_pro_record": record, "frames": frame_records,
        "literal_code_excerpts": selections, "posterior_from_replayed_trace": posterior,
        "checks": {"original_direct_is_exact_single_letter": True, "saved_prediction_matches_raw": True,
                   "not_in_unresolved69": True, "spatial_trace_replay_exact": True,
                   "social_trace_replay_exact": replayed_social == record["social_trace"],
                   "social_trace_original_fields_match_after_documented_default_field_removal": True,
                   "original8_frame_hashes_verified": True, "actual_pro_wire_request_available": False,
                   "actor_identity_from_supplied_text_not_face_recognition": True,
                   "question_premise_is_not_visual_inference": True, "no_win_loss_badges_in_panel": True},
        "visual_review": {"reviewed_contact_sheet": True, "supported": "An actor appears in a kitchen and changes position across the eight frames.",
                          "not_visually_verified": ["Actor name John", "cupcake inside microwave", "dish bowl inside fridge", "John's belief"],
                          "sources_for_these_claims": "Supplied scene/actions or explicit query premise, not pixels alone."},
        "interpretation": "Historical static frame+text compilation with candidate-conditioned rule/posterior readout, not causal evidence for cross-layer links or online belief updating."})
    write(OUT / "selection_ledger_600.json", ledger)
    write(OUT / "selection.json", {"selection": "Post-hoc qualitative example: first index with explicit consistent Direct answer, Direct wrong under archived labels, SSC correct, no dense refinement.",
          "selected": index, "eligible_all": sum(r["candidate"] for r in ledger), "eligible_without_dense": len(eligible),
          "not_an_evaluation_sample": True, "rejected_old_index0": "Pro raw text has no explicit final choice; remove Direct incorrect badge.",
          "actual_pro_request_mapping_not_independently_verified": True})
    # Review the available MA candidate only; do not assume it is the user's panel(b).
    ma_path = ROOT / "reports/cross_environment_casebook_20260923/ma_egoqa/index_1038/case.json"
    ma = json.loads(ma_path.read_text())
    for spec in ma["media"]:
        assert sha(ROOT / spec["path"]) == spec["sha256"]
    for spec in ma["preview_frames"]:
        assert sha(ma_path.parent / spec["preview_file"]) == spec["preview_sha256"]
    ids = set(ma["cited_evidence"])
    for line in ma["compiled"]["social_trace"].splitlines():
        assert set(json.loads(line)["values"].get("evidence_ids", [])) <= ids
    write(OUT / "panel_b_review.json", {
        "candidate_checked": "MA-EgoQA1038 from existing casebook, NOT confirmed to be the actual manuscript panel(b)",
        "model_scope": ma["scope"], "source_case_sha256": sha(ma_path),
        "input_clip_hashes_checked": len(ma["media"]), "preview_hashes_checked": len(ma["preview_frames"]),
        "social_references_resolve": True,
        "reported_speaker_attribution": "Generated S2 assigns a question to Tasha; S3 assigns a reply to Jake.",
        "independent_speaker_or_transcript_verification": False,
        "all_three_saved_answers": "C", "gold": "C",
        "cannot_infer": ["speaker identity from camera wearer name", "quoted speech from an image", "causal evidence-link benefit from correct final answer"],
        "action": "Do not use speech balloons or person-grounded correction narrative as verified fact. This candidate is Flash, so do not substitute it into a Pro-only figure. Need actual panel(b) source to finish that audit."})
    render(folder, frame_records, posterior)
    manifest = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.json"}
    write(OUT / "SHA256SUMS.json", manifest)
    print("PASS: Pro302 original raw/code/trace/frame audit; replacement assets rendered; no other panels changed.")


def render(folder, frames, posterior):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image, ImageDraw
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    contact = Image.new("RGB", (1280, 420), "white")
    draw = ImageDraw.Draw(contact)
    for j, spec in enumerate(frames):
        with Image.open(OUT / spec["file"]) as image:
            im = image.convert("RGB"); im.thumbnail((310, 175))
        x, y = (j%4)*320, (j//4)*210
        contact.paste(im, (x,y))
        draw.text((x+5,y+181), f"Input {j+1}; source frame {spec['frame_index']}", fill="black")
    contact.save(folder / "frames_contact_sheet.png")
    fig = plt.figure(figsize=(15, 5), facecolor="white")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(.025,.96,"(a) MMToM — physical fact, attributed premise, and saved readout", fontsize=16, weight="bold", va="top")
    ax.text(.025,.895,"Historical Pro • index 302 • episode 14 • static RGB-frame + text input", fontsize=10, color="#526070")
    xs = [.025,.36,.695]
    for x, title in zip(xs,["1  INPUT", "2  STORED REPRESENTATION", "3  HISTORICAL READOUT"]):
        ax.add_patch(FancyBboxPatch((x,.13),.29 if x == xs[-1] else .305,.69,boxstyle="round,pad=.009,rounding_size=.018",fc="#f7f9fc",ec="#c8d2df"))
        ax.text(x+.014,.778,title,fontsize=12,weight="bold",color="#203650")
    for x in [.337,.672]:
        ax.annotate("",xy=(x+.016,.47),xytext=(x-.009,.47),arrowprops={"arrowstyle":"->","color":"#75849a","lw":1.8})
    for j in range(2):
        image_ax = fig.add_axes([.039+j*.143,.51,.136,.23]); image_ax.axis("off")
        image_ax.imshow(Image.open(OUT/frames[[3,7][j]]["file"]))
        image_ax.set_title(f"Source frame {frames[[3,7][j]]['frame_index']}",fontsize=8)
    ax.text(.04,.46,"Supplied action history",fontsize=10,weight="bold")
    ax.text(.04,.414,"John opens/closes cabinet 4,\nthen approaches the microwave.",fontsize=10,linespacing=1.5,va="top")
    ax.text(.04,.293,"Question premise (given)",fontsize=10,weight="bold",color="#815324")
    ax.text(.04,.248,"John thinks there is no cupcake\ninside the microwave.",fontsize=10,linespacing=1.5,va="top")
    ax.text(.375,.694,"Physical fact · supplied scene text",fontsize=10,weight="bold",color="#246087")
    ax.text(.375,.632,"cupcake  →  inside microwave",fontsize=12)
    ax.text(.375,.52,"Attributed belief · query premise",fontsize=10,weight="bold",color="#815324")
    ax.text(.375,.456,"John  →  no cupcake in microwave",fontsize=11)
    ax.text(.375,.327,"Different sources; not a contradiction\ninside a single world-state record.",fontsize=10,linespacing=1.5,va="top")
    ax.text(.375,.205,"The belief is given, not visually inferred.",fontsize=9,color="#526070")
    ax.text(.71,.691,"Candidate-conditioned rule + posterior",fontsize=10,weight="bold")
    ax.text(.71,.622,f"A  dish bowl       {posterior['A']:.4f}",fontsize=12)
    ax.text(.71,.553,f"B  cupcake         {posterior['B']:.4f}",fontsize=12)
    ax.text(.71,.427,"Archived Direct response:  B\nArchived SSC readout:        A\nBenchmark label:                 A",fontsize=11,linespacing=1.75,va="top")
    ax.text(.71,.205,"Saved labels shown without win/loss badges.",fontsize=9,color="#526070")
    ax.text(.025,.055,"Record-level illustration, not a link ablation or online update. Frame indices are not seconds; actor name and hidden contents come from text.",fontsize=9,color="#526070")
    for suffix in ("svg","pdf","png"):
        fig.savefig(folder/f"panel_a_input_state_readout.{suffix}",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-export", action="store_true", help="Redraw solely from the public case package; no local benchmark needed")
    args = parser.parse_args()
    if args.from_export:
        case = json.loads((OUT / "mmtom_pro_0302/case.json").read_text())
        render(OUT / "mmtom_pro_0302", case["frames"], case["posterior_from_replayed_trace"])
        write(OUT / "SHA256SUMS.json", {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.json"})
    else:
        main()
