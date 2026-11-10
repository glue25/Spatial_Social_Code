import asyncio
import base64
import functools
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from sam3.model_builder import build_sam3_multiplex_video_predictor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
_PREDICTOR: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"

DEFAULT_SAM31_CHECKPOINT = Path(
    "/pfs/zhengshenghe/code/WAM/WAM-DreamZero/data_processing/seg/model/"
    "sam3.1/sam3.1_multiplex.pt"
)

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

# --- Helper Functions ---


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor-like object to numpy array."""
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask(mask: np.ndarray) -> str:
    # Pack boolean mask to bytes (uint8) then base64
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 bytes."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")


# --- Request/Response Models ---


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]  # [x, y] — JSON arrays, not tuples


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: list[int]  # [num_masks, H, W]
    masks_dtype: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


# --- Core Logic ---


def _start_image_session(pil_image: Image.Image) -> str:
    """Create a short-lived SAM 3.1 video session for one PIL image."""
    # The generic predictor's start_session currently passes an
    # offload_state_to_cpu argument that the SAM 3.1 model does not accept.
    # Initialize the state directly, matching Meta's SAM 3.1 frame-list API.
    state = _PREDICTOR.model.init_state(
        resource_path=[pil_image],
        offload_video_to_cpu=False,
        async_loading_frames=False,
    )

    # Preserve the source resolution in output masks. The PIL-list loader may
    # otherwise retain the model's internal square resolution here.
    state["orig_height"] = pil_image.height
    state["orig_width"] = pil_image.width

    session_id = str(uuid.uuid4())
    now = time.time()
    _PREDICTOR._all_inference_states[session_id] = {
        "state": state,
        "session_id": session_id,
        "start_time": now,
        "last_use_time": now,
    }
    return session_id


def _close_session(session_id: str) -> None:
    _PREDICTOR.handle_request(
        {
            "type": "close_session",
            "session_id": session_id,
            "run_gc_collect": False,
        }
    )


def _do_segment(pil_image: Image.Image, text_prompt: str):
    """Blocking SAM 3.1 text-prompt segmentation (runs on GPU thread)."""
    session_id = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            session_id = _start_image_session(pil_image)
            response = _PREDICTOR.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": text_prompt,
                }
            )
            output = response["outputs"]
    finally:
        if session_id is not None:
            _close_session(session_id)

    masks_tensor = output.get("out_binary_masks")
    boxes_tensor = output.get("out_boxes_xywh")
    scores_tensor = output.get("out_probs")

    if masks_tensor is None or boxes_tensor is None:
        return SegmentResponse(results=[])

    masks_np = _to_numpy(masks_tensor)
    boxes_np = _to_numpy(boxes_tensor)
    scores_np = (
        _to_numpy(scores_tensor)
        if scores_tensor is not None
        else np.ones(len(masks_np), dtype=np.float32)
    )

    # Squeeze masks if needed: (N, 1, H, W) -> (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    results_data = []
    num_preds = len(scores_np)

    for i in range(num_preds):
        mask = masks_np[i] > 0  # Boolean mask
        # SAM 3.1 returns normalized [x, y, width, height]. Preserve this
        # server's original response contract: pixel [x1, y1, x2, y2].
        x, y, width, height = boxes_np[i]
        box = [
            float(x * pil_image.width),
            float(y * pil_image.height),
            float((x + width) * pil_image.width),
            float((y + height) * pil_image.height),
        ]
        score = float(scores_np[i])

        results_data.append(
            MaskData(
                mask_base64=encode_mask(mask),
                shape=mask.shape,
                box=box,
                score=score,
                label=text_prompt,
            )
        )

    # Sort by score descending
    results_data.sort(key=lambda x: x.score, reverse=True)

    return SegmentResponse(results=results_data)


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


def _do_segment_point(pil_image: Image.Image, point_coords_tuple: tuple[float, float]):
    """Blocking SAM 3.1 foreground-point segmentation."""
    session_id = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            session_id = _start_image_session(pil_image)
            response = _PREDICTOR.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": [list(point_coords_tuple)],
                    "point_labels": [1],
                    "obj_id": 1,
                    "rel_coordinates": False,
                }
            )
            output = response.get("outputs") or {}
            if len(output.get("out_binary_masks", [])) == 0:
                for propagated in _PREDICTOR.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "start_frame_index": 0,
                        "max_frame_num_to_track": 1,
                    }
                ):
                    if propagated["frame_index"] == 0:
                        output = propagated["outputs"]
    finally:
        if session_id is not None:
            _close_session(session_id)

    masks_np = np.asarray(output.get("out_binary_masks", []))
    scores_np = np.asarray(output.get("out_probs", []))

    if masks_np.size and scores_np.size == 0:
        scores_np = np.ones(masks_np.shape[0], dtype=np.float32)

    if masks_np.size == 0 or scores_np.size == 0:
        return PointPromptResponse(
            scores=[],
            masks_base64="",
            masks_shape=(0, 0, 0),
            masks_dtype="float32",
        )

    # Sort by score descending
    sort_idx = np.argsort(scores_np)[::-1]
    masks_np = masks_np[sort_idx]
    scores_np = scores_np[sort_idx]

    return PointPromptResponse(
        scores=scores_np.astype(float).tolist(),
        masks_base64=encode_array(masks_np),
        masks_shape=tuple(masks_np.shape),
        masks_dtype=str(masks_np.dtype),
    )


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PREDICTOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment_point, pil_image, req.point_coords)
    except Exception as e:
        logger.error(f"Point prompt inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Point prompt inference failed: {e}")


@app.get("/health")
async def health():
    return {
        "ready": _PREDICTOR is not None,
        "model": "SAM 3.1 Multiplex",
        "device": _DEVICE,
    }


def main(
    device: str = "cuda",
    port: int = 8114,
    host: str = "127.0.0.1",
    checkpoint_path: Path = DEFAULT_SAM31_CHECKPOINT,
):
    global _MODEL, _PREDICTOR, _DEVICE

    _DEVICE = device

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM 3.1 checkpoint not found: {checkpoint_path}")
    if "cuda" not in device or not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 Multiplex currently requires an available CUDA GPU")

    # Setup torch settings for Ampere+ GPUs as recommended
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set the default CUDA device so all tensor allocations inside
        # build_sam3_image_model and Sam3Processor use the correct device.
        # Without this, internal tensors default to cuda:0 while model
        # weights are on the specified device, causing device mismatch errors.
        if "cuda" in device:
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_idx)

    logger.info("Loading SAM 3.1 Multiplex checkpoint from %s", checkpoint_path)
    try:
        _PREDICTOR = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        _MODEL = _PREDICTOR.model
        # The published SAM 3.1 checkpoint uses the separate detector score.
        _MODEL.detector.supervise_joint_box_scores = False
    except Exception as e:
        logger.error(f"Error building SAM 3.1 model: {e}")
        raise

    logger.info("SAM 3.1 loaded on %s. Starting server...", device)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
