#!/usr/bin/env python3
"""
Eye Status API — 눈 크롭 이미지를 받아 open/closed 상태를 판정하는 무상태 추론 API.

클라이언트(웹캠 캡처, 눈 크롭, 알람 재생, 타이머 로직)와 역할이 분리되어 있습니다.
이 서버는 이미지를 받아 분류 결과만 반환합니다.
"""

import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "dima806/closed_eyes_image_detection"
CLOSED_CONFIDENCE = 0.60
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    _state["processor"] = processor
    _state["model"] = model
    yield
    _state.clear()


app = FastAPI(title="Eye Status API", lifespan=lifespan)


class EyeStatusResponse(BaseModel):
    left_closed_prob: float
    right_closed_prob: float
    avg_closed_prob: float
    is_closed: bool
    threshold: float
    device: str


def _closed_prob(image: Image.Image) -> float:
    processor = _state["processor"]
    model = _state["model"]
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    closed_p = 0.0
    for idx, label in model.config.id2label.items():
        if "close" in label.lower():
            closed_p = max(closed_p, probs[idx].item())
    return closed_p


async def _load_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail=f"invalid image: {upload.filename}")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/v1/eye-status", response_model=EyeStatusResponse)
async def eye_status(
    left_eye: UploadFile = File(...),
    right_eye: UploadFile = File(...),
):
    left_img = await _load_image(left_eye)
    right_img = await _load_image(right_eye)

    left_p = await run_in_threadpool(_closed_prob, left_img)
    right_p = await run_in_threadpool(_closed_prob, right_img)
    avg_p = (left_p + right_p) / 2

    return EyeStatusResponse(
        left_closed_prob=left_p,
        right_closed_prob=right_p,
        avg_closed_prob=avg_p,
        is_closed=avg_p >= CLOSED_CONFIDENCE,
        threshold=CLOSED_CONFIDENCE,
        device=DEVICE,
    )
