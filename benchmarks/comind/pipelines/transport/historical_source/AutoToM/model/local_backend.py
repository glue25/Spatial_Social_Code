"""Local Transformers backend used by the AutoToM Qwen adaptation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_PATH = os.getenv(
    "SPATIAL_SOCIAL_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"
)
LOCAL_FILES_ONLY = os.getenv("SPATIAL_SOCIAL_LOCAL_FILES_ONLY", "0") == "1"
CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "local_qwen3vl.jsonl"

_processor = None
_model = None
_cache: dict[str, object] | None = None
_image_paths: tuple[Path, ...] = ()
_image_signature = ""
_image_inputs: tuple[Image.Image, ...] = ()


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(
            MODEL_PATH, local_files_only=LOCAL_FILES_ONLY
        )
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=LOCAL_FILES_ONLY,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
    return _processor, _model


def _load_cache() -> dict[str, object]:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_PATH.exists():
            with CACHE_PATH.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        item = json.loads(line)
                        _cache[item["key"]] = item["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue
    return _cache


def set_image_context(paths: Sequence[str | Path]) -> None:
    """Attach the current MMToM question's sampled RGB frames to all calls."""
    global _image_paths, _image_signature, _image_inputs

    resolved = tuple(Path(path).resolve() for path in paths)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MMToM RGB frame(s): {missing}")

    identities = [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in resolved
    ]
    signature = hashlib.sha256(
        json.dumps(identities, sort_keys=True).encode()
    ).hexdigest()
    if signature == _image_signature:
        return

    images = []
    for path in resolved:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    _image_paths = resolved
    _image_inputs = tuple(images)
    _image_signature = signature


def clear_image_context() -> None:
    global _image_paths, _image_signature, _image_inputs
    _image_paths = ()
    _image_inputs = ()
    _image_signature = ""


def image_context_paths() -> tuple[Path, ...]:
    return _image_paths


def _key(kind: str, prompt: str, extra: str = "") -> str:
    payload = f"{kind}\0{prompt}\0{extra}"
    if _image_signature:
        payload += f"\0images:{_image_signature}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _store(key: str, value: object) -> None:
    cache = _load_cache()
    cache[key] = value
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _inputs(prompt: str):
    processor, _ = _load_model()
    if _image_inputs:
        content = [
            {"type": "image", "image": str(path)} for path in _image_paths
        ]
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    processor_kwargs = {"text": [text], "return_tensors": "pt"}
    if _image_inputs:
        processor_kwargs["images"] = list(_image_inputs)
        processor_kwargs["padding"] = True
    return processor(**processor_kwargs).to("cuda")


def generate(prompt: str, max_new_tokens: int = 512) -> str:
    cache_key = _key("generate", prompt, str(max_new_tokens))
    cached = _load_cache().get(cache_key)
    if isinstance(cached, str):
        return cached

    processor, model = _load_model()
    inputs = _inputs(prompt)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=min(max_new_tokens, 512),
        )
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    if "</think>" in result:
        result = result.split("</think>", 1)[1].strip()
    _store(cache_key, result)
    return result


def choice_probabilities(prompt: str, choices: Iterable[str]) -> dict[str, float]:
    choice_list = list(choices)
    cache_key = _key("choices", prompt, "".join(choice_list))
    cached = _load_cache().get(cache_key)
    if isinstance(cached, dict):
        return {str(k): float(v) for k, v in cached.items()}

    processor, model = _load_model()
    inputs = _inputs(prompt)
    token_ids = []
    for choice in choice_list:
        encoded = processor.tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Choice {choice!r} is not a single token: {encoded}")
        token_ids.append(encoded[0])

    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    selected = logits[torch.tensor(token_ids, device=logits.device)]
    values = torch.softmax(selected, dim=0).cpu().tolist()
    result = dict(zip(choice_list, values, strict=True))
    _store(cache_key, result)
    return result


def probability_a(prompt: str) -> float:
    return choice_probabilities(prompt, ["A", "B"])["A"]
