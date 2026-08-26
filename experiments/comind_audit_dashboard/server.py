#!/usr/bin/env python3
"""Read-only CoMind answer audit dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import random
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = (ROOT / "workdirs/comind_full_recording").resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V2 = load_module("comind_v2_dashboard", ROOT / "experiments/comind_optimization_v2/comind_v2.py")
PILOT = load_module("comind_weak_pilot_dashboard", ROOT / "experiments/comind_weak_skill_v1/run_paired_pilot.py")

TASK_TEXT = {
    "joint_attention": "预测两位参与者共同注意的物体：左右视角 bbox、触发 cue，以及一级物体类别。",
    "scoia": "根据 helper 在预测时刻之前的观察，预测未来 10 秒内受社会线索影响而执行的动作、交互物体和 cue。",
    "handover": "预测即将发生的递交：物体、方向、发起者、发起方式，以及距离递交发生的时间。",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> str:
    return "|".join((str(row["task"]), str(row["recording_id"]), str(row["sample_id"])))


class Store:
    def __init__(self) -> None:
        self.manifest = {key(row): row for row in read_jsonl(ROOT / "workdirs/comind_optimization_v2/gpu_grounding_full_manifest.jsonl")}
        self.routed = {key(row): row for row in read_jsonl(ROOT / "workdirs/comind_optimization_v2/full_train_oof_predictions.jsonl")}
        self.final = {key(row): row for row in read_jsonl(ROOT / "workdirs/comind_optimization_v2/weak_fields_nested_predictions.jsonl")}
        self.grounding = {key(row): row for row in read_jsonl(ROOT / "workdirs/comind_optimization_v2/two_stage_grounding_full_predictions.jsonl")}
        self.annotations = V2.load_annotations(ROOT / "external_datasets/comind/annotations")

    def pilot(self) -> dict[str, dict]:
        return {key(row): row for row in read_jsonl(ROOT / "workdirs/comind_weak_skill_v1/pilot_predictions.jsonl")}

    def gold(self, item_key: str) -> dict:
        task, recording, sample = item_key.split("|", 2)
        return self.annotations.get(task, {}).get(recording, {}).get(sample, {})

    def canonical_gold(self, item_key: str) -> dict:
        task = item_key.split("|", 1)[0]
        gold = self.gold(item_key)
        return {
            metric: V2.ground_truth_value(task, metric, gold)
            for metric in V2.FIELD_TO_PREDICTION[task]
        }

    def metrics(self, item_key: str, dataset: str) -> list[str]:
        task = item_key.split("|", 1)[0]
        if dataset == "pilot":
            return list(PILOT.METRICS.get(task, ()))
        return list(V2.FIELD_TO_PREDICTION[task])

    def score_pair(self, item_key: str, dataset: str, metric: str) -> tuple[bool, bool] | None:
        if dataset == "pilot":
            row = self.pilot().get(item_key)
            if not row or "scores" not in row or metric not in row["scores"]["baseline"]:
                return None
            return bool(row["scores"]["baseline"][metric]), bool(row["scores"]["temporal_social_skill"][metric])
        routed, final = self.routed.get(item_key), self.final.get(item_key)
        if not routed or not final:
            return None
        baseline = routed.get("scores", {}).get("direct", {}).get(metric)
        method = final.get("scores", {}).get(metric)
        if baseline is None or method is None:
            return None
        return bool(baseline), bool(method)

    def list_items(self, query: dict[str, list[str]]) -> dict:
        dataset = query.get("dataset", ["pilot"])[0]
        task = query.get("task", [""])[0]
        metric = query.get("metric", [""])[0]
        outcome = query.get("outcome", ["any"])[0]
        search = query.get("search", [""])[0].strip().lower()
        keys = list(self.pilot()) if dataset == "pilot" else list(set(self.routed) & set(self.final) & set(self.manifest))
        keys.sort()
        filtered = []
        for item_key in keys:
            if task and not item_key.startswith(task + "|"):
                continue
            if search and search not in item_key.lower():
                continue
            if metric and outcome != "any":
                pair = self.score_pair(item_key, dataset, metric)
                if pair is None:
                    continue
                baseline, method = pair
                wanted = {
                    "method_gain": (False, True),
                    "method_regression": (True, False),
                    "both_correct": (True, True),
                    "both_wrong": (False, False),
                }.get(outcome)
                if wanted and pair != wanted:
                    continue
            filtered.append(item_key)
        if query.get("shuffle", ["0"])[0] == "1":
            random.Random(20260826).shuffle(filtered)
        return {"dataset": dataset, "count": len(filtered), "keys": filtered}

    def detail(self, item_key: str, dataset: str) -> dict:
        manifest = self.manifest[item_key]
        routed = self.routed.get(item_key, {})
        final = self.final.get(item_key, {})
        grounding = self.grounding.get(item_key, {})
        pilot = self.pilot().get(item_key)
        images = [
            {**item, "url": "/image?path=" + urllib.parse.quote(item["path"])}
            for item in manifest.get("images", [])
        ]
        prompts = {}
        if pilot:
            prompts = {
                "baseline": PILOT.row_prompt(manifest, "baseline"),
                "temporal_social_skill": PILOT.row_prompt(manifest, "temporal_social_skill"),
            }
        return {
            "key": item_key,
            "dataset": dataset,
            "task": manifest["task"],
            "question": TASK_TEXT[manifest["task"]],
            "prediction_time": manifest.get("prediction_time"),
            "transcript": manifest.get("transcript_10s", ""),
            "fold": manifest.get("recording_fold"),
            "images": images,
            "gold_canonical": self.canonical_gold(item_key),
            "gold_raw": self.gold(item_key),
            "answers": {
                "direct_baseline": routed.get("direct"),
                "spatial_social": routed.get("spatial_social"),
                "provenance_router": routed.get("provenance_router"),
                "grounding": grounding.get("prediction"),
                "final_combined": final.get("prediction"),
                "fair_pilot_baseline": (pilot or {}).get("predictions", {}).get("baseline"),
                "fair_pilot_skill": (pilot or {}).get("predictions", {}).get("temporal_social_skill"),
            },
            "scores": {
                "historical": routed.get("scores"),
                "final_combined": final.get("scores"),
                "fair_pilot": (pilot or {}).get("scores"),
            },
            "prompts": prompts,
            "fairness": {
                "checkpoint": "Qwen3-VL-8B-Instruct",
                "same_samples": True,
                "same_images_and_transcript": True,
                "same_label_ontology": True,
                "calls_per_condition": 1,
                "decoding": "greedy",
                "max_new_tokens": 256,
                "selection": "48 SCOIA + 48 Handover; each task uses 48 distinct recordings; deterministic pre-run hash ordering",
            },
        }


HTML = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoMind 逐题审计</title><style>
:root{--bg:#0b1020;--panel:#121a2d;--line:#263451;--text:#e9efff;--muted:#94a3c4;--good:#35d07f;--bad:#ff6978;--accent:#7aa2ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}header{position:sticky;top:0;z-index:4;background:#0b1020ee;border-bottom:1px solid var(--line);padding:14px 20px}h1{font-size:20px;margin:0 0 10px}.controls{display:flex;gap:8px;flex-wrap:wrap}select,input,button{background:#17223b;color:var(--text);border:1px solid #344669;border-radius:7px;padding:7px 10px}button{cursor:pointer}.status{color:var(--muted);margin-left:auto;padding:7px}.wrap{max-width:1600px;margin:auto;padding:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}.muted{color:var(--muted)}.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.frame img{width:100%;max-height:260px;object-fit:contain;background:#050810;border-radius:6px}.frame div{font-size:12px;color:var(--muted)}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.answer{border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}.answer h3{margin:0 0 7px;font-size:14px;color:var(--accent)}pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.45 ui-monospace,monospace}.score-grid{display:flex;gap:8px;flex-wrap:wrap}.pill{border-radius:99px;padding:4px 8px;background:#27334c}.ok{color:var(--good)}.no{color:var(--bad)}details summary{cursor:pointer;color:var(--accent)}
</style></head><body><header><h1>CoMind 逐题表现审计</h1><div class="controls">
<select id="dataset"><option value="pilot">公平弱字段 pilot（96）</option><option value="full">全量 OOF（6673）</option></select>
<select id="task"><option value="">全部任务</option><option value="joint_attention">Joint Attention</option><option value="scoia">SCOIA</option><option value="handover">Handover</option></select>
<select id="metric"><option value="">全部字段</option></select><select id="outcome"><option value="any">全部结果</option><option value="method_gain">我们对 / baseline 错</option><option value="method_regression">我们错 / baseline 对</option><option value="both_correct">两边都对</option><option value="both_wrong">两边都错</option></select>
<input id="search" placeholder="recording / sample ID"><button id="apply">筛选</button><button id="random">固定随机顺序</button><button id="prev">上一题</button><button id="next">下一题</button><span class="status" id="status"></span></div></header><main class="wrap">
<section class="card"><b>公平性协议</b><div id="fairness" class="meta muted"></div></section><section id="content"></section></main>
<script>
const metrics={pilot:{'':['cue_type','action_verb','delivery_flow','initiator','initiation_type','tth_within_0_25s'],scoia:['cue_type','action_verb'],handover:['delivery_flow','initiator','initiation_type','tth_within_0_25s'],joint_attention:[]},full:{'':['left_iou_at_0_5','right_iou_at_0_5','iou_at_0_5','cue_type','category_l1','action_verb','action_noun_l1','delivery_flow','initiator','initiation_type','tth_within_0_25s'],joint_attention:['left_iou_at_0_5','right_iou_at_0_5','cue_type','category_l1'],scoia:['iou_at_0_5','cue_type','action_verb','action_noun_l1'],handover:['iou_at_0_5','delivery_flow','initiator','initiation_type','category_l1','tth_within_0_25s']}};let keys=[],idx=0,shuffle=0;
const $=x=>document.getElementById(x), esc=x=>JSON.stringify(x,null,2);function updateMetrics(){let old=$('metric').value,ds=$('dataset').value,t=$('task').value;$('metric').innerHTML='<option value="">全部字段</option>'+metrics[ds][t].map(x=>`<option>${x}</option>`).join('');if(metrics[ds][t].includes(old))$('metric').value=old}
async function loadList(){updateMetrics();let q=new URLSearchParams({dataset:$('dataset').value,task:$('task').value,metric:$('metric').value,outcome:$('outcome').value,search:$('search').value,shuffle:String(shuffle)});let d=await fetch('/api/items?'+q).then(r=>r.json());keys=d.keys;idx=0;await show()}
function box(title,obj){if(obj==null)return '';return `<div class="answer"><h3>${title}</h3><pre>${esc(obj)}</pre></div>`}function scoreBox(scores){if(!scores)return '';let out=[];for(let [group,vals] of Object.entries(scores)){if(!vals)continue;let inner=[];for(let [m,v] of Object.entries(vals)){if(v&&typeof v==='object'){for(let [m2,v2] of Object.entries(v))inner.push(`<span class="pill ${v2?'ok':'no'}">${m2}: ${v2?'✓':'✗'}</span>`)}else inner.push(`<span class="pill ${v?'ok':'no'}">${m}: ${v?'✓':'✗'}</span>`)}out.push(`<div><b>${group}</b><div class="score-grid">${inner.join('')}</div></div>`)}return out.join('<br>')}
async function show(){if(!keys.length){$('status').textContent='0 条';$('content').innerHTML='<section class="card">没有匹配题目</section>';return}$('status').textContent=`${idx+1} / ${keys.length}`;let d=await fetch('/api/item?dataset='+$('dataset').value+'&key='+encodeURIComponent(keys[idx])).then(r=>r.json());$('fairness').innerHTML=Object.entries(d.fairness).map(([k,v])=>`<div>${k}: <b>${v}</b></div>`).join('');let frames=d.images.map(x=>`<div class="frame"><img loading="lazy" src="${x.url}"><div>${x.role} · ${x.kind} · ${Number(x.time).toFixed(3)}s</div></div>`).join('');let answers=Object.entries(d.answers).map(([k,v])=>box(k,v)).join('');let prompts=Object.entries(d.prompts).map(([k,v])=>`<details><summary>${k} exact prompt</summary><pre>${v}</pre></details>`).join('');$('content').innerHTML=`<section class="card"><div class="meta"><div><b>${d.task}</b><br><span class="muted">${d.key}</span></div><div>fold: ${d.fold}<br>prediction: ${d.prediction_time}s</div></div><h3>题目</h3><p>${d.question}</p><h3>预测前 transcript</h3><pre>${d.transcript||'&lt;none&gt;'}</pre>${prompts}</section><section class="card"><h2>输入帧（按时间顺序）</h2><div class="frames">${frames}</div></section><section class="card"><h2>Gold</h2><div class="answers">${box('canonical gold',d.gold_canonical)}${box('raw annotation',d.gold_raw)}</div></section><section class="card"><h2>具体回答</h2><div class="answers">${answers}</div></section><section class="card"><h2>逐字段判分</h2>${scoreBox(d.scores)}</section>`;window.scrollTo({top:0,behavior:'smooth'})}
$('apply').onclick=()=>{shuffle=0;loadList()};$('random').onclick=()=>{shuffle=1;loadList()};$('prev').onclick=()=>{if(keys.length){idx=(idx-1+keys.length)%keys.length;show()}};$('next').onclick=()=>{if(keys.length){idx=(idx+1)%keys.length;show()}};$('dataset').onchange=loadList;$('task').onchange=loadList;loadList();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if parsed.path == "/api/items":
                return self.send_bytes(json.dumps(self.store.list_items(query)).encode(), "application/json")
            if parsed.path == "/api/item":
                item_key = query.get("key", [""])[0]
                dataset = query.get("dataset", ["pilot"])[0]
                return self.send_bytes(json.dumps(self.store.detail(item_key, dataset)).encode(), "application/json")
            if parsed.path == "/image":
                path = Path(query.get("path", [""])[0]).resolve()
                if not path.is_relative_to(IMAGE_ROOT) or not path.is_file():
                    return self.send_bytes(b"not found", "text/plain", 404)
                return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_bytes(b"not found", "text/plain", 404)
        except Exception as error:
            self.send_bytes(json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json", 500)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    Handler.store = Store()
    print(f"CoMind audit dashboard: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
