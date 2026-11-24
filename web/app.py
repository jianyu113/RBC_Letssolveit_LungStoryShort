"""Serve the LungStoryShort prototype and the fixed, frozen model pair."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser

from .inference import FUSION, MODEL_NAMES, ModelService
from .access import GroupAccessMiddleware


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WEIGHTS = REPO.parent.parent / "model_weights"
MAX_BYTES = 32 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    app.state.service = None
    app.state.detail = "Models are not loaded."
    app.state.lock = asyncio.Lock()
    try:
        app.state.service = ModelService(
            os.environ.get("LSS_DETECTOR", str(WEIGHTS / "baseline44_epoch6.pt")),
            os.environ.get("LSS_CLASSIFIER", str(WEIGHTS / "resnet18_baseline_epoch2.pt")),
            device=os.environ.get("LSS_DEVICE", "cpu"),
        )
        app.state.detail = "Both verified checkpoints are ready."
    except Exception as error:
        logging.exception("Model loading failed")
        app.state.detail = f"Model loading failed: {error}"
    yield


app = FastAPI(title="LungStoryShort", lifespan=lifespan)
app.add_middleware(GroupAccessMiddleware)


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
    )
    return response


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ready": app.state.service is not None, "detail": app.state.detail,
            "models": MODEL_NAMES, "fusion": FUSION}


def require_models():
    if app.state.service is None:
        raise HTTPException(503, app.state.detail)
    return app.state.service


async def predict(content, name, kind):
    service = require_models()
    try:
        async with app.state.lock:
            result = await run_in_threadpool(service.predict, content, name, kind)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logging.exception("Image inference failed")
        raise HTTPException(500, "Inference failed. Check the server log.") from error
    result["id"] = hashlib.sha256(content).hexdigest()[:16]
    result["file_name"] = name
    return result


@app.post("/api/analyze")
async def analyze(request: Request):
    require_models()
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(415, "Upload images using multipart/form-data.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Upload limit is 32 MB per request.")
        body.extend(chunk)

    async def stream():
        yield bytes(body)

    # The bounded request fits in memory, so uploads do not spill to disk.
    parser = MultiPartParser(request.headers, stream(), max_files=20, max_fields=1)
    parser.spool_max_size = MAX_BYTES
    try:
        form = await parser.parse()
    except MultiPartException as error:
        raise HTTPException(400, str(error)) from error
    try:
        kind = form.get("input_kind")
        files = form.getlist("files")
        if kind not in {"dicom", "prepared_png"} or not files:
            raise HTTPException(422, "Choose DICOM or prepared PNG and at least one file.")
        if any(not hasattr(file, "read") for file in files):
            raise HTTPException(422, "The files field must contain uploaded files.")
        results = [await predict(await file.read(), Path(file.filename or "image").name, kind)
                   for file in files]
        return {"results": results, "fusion": FUSION}
    finally:
        await form.close()


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
