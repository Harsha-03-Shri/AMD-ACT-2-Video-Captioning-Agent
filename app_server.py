"""HTTP API entry point for running the captioning agent as a service on Render.

Job-based contract with the Replit UI (or any caller):

    POST /process
        headers: {"X-API-Key": "<API_AUTH_TOKEN>"}
        body:    {"video_url": "<downloadable url>", "styles": ["casual", "formal"]}
        response (202): {"job_id": "...", "status": "pending"}

    GET /status/{job_id}
        headers: {"X-API-Key": "<API_AUTH_TOKEN>"}
        response:
            pending:  {"job_id": "...", "status": "pending"}
            done:     {"job_id": "...", "status": "done", "captions": {"casual": "...", "formal": "..."}}
            error:    {"job_id": "...", "status": "error", "error": "<message>"}

    GET /health
        (no auth required)
        response: {"status": "ok"}

Auth: every request to /process and /status/{job_id} must include header
      X-API-Key: <value of API_AUTH_TOKEN env var>
      Missing/incorrect key -> 401.

This reuses the same provider pools / download / keyframe / sanitize logic as the
batch pipeline in main.py, just driven per-request instead of from tasks.json.

Jobs are tracked in an in-memory dict. This is fine for a single Render instance;
if you ever scale to multiple instances, swap `jobs` for shared storage (e.g. Redis).
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import FALLBACK_CAPTION
from app.downloader import download_video
from app.keyframe import extract_keyframes
from app.providers import (
    GenerationError,
    UploadError,
    build_caption_providers,
    build_description_providers,
    load_model_config,
)
from app.sanitize import sanitize_caption

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Populated at startup, shared across requests
state: dict = {"desc_providers": [], "caption_providers": []}

# In-memory job store: job_id -> job dict
jobs: dict[str, dict] = {}

API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_AUTH_TOKEN:
        logger.warning(
            "API_AUTH_TOKEN is not set — /process and /status will reject all requests. "
            "Set API_AUTH_TOKEN in the environment to enable access."
        )
    config = load_model_config()
    state["desc_providers"] = build_description_providers(config)
    state["caption_providers"] = build_caption_providers(config)
    logger.info(
        "Server ready: %d description providers, %d caption providers",
        len(state["desc_providers"]), len(state["caption_providers"]),
    )
    yield
    state["desc_providers"].clear()
    state["caption_providers"].clear()


app = FastAPI(title="Video Captioning Agent", lifespan=lifespan)


# ---------- Auth ----------

async def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_AUTH_TOKEN or x_api_key != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ---------- Schemas ----------

class JobStatus(str, Enum):
    pending = "pending"
    done = "done"
    error = "error"


class ProcessRequest(BaseModel):
    video_url: str
    styles: list[str] = Field(default_factory=lambda: ["default"])


class ProcessAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    captions: dict[str, str] | None = None
    error: str | None = None


# ---------- Health (no auth) ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Submit job ----------

@app.post("/process", response_model=ProcessAccepted, status_code=202, dependencies=[Depends(require_api_key)])
async def process_video(req: ProcessRequest):
    if not state["desc_providers"] or not state["caption_providers"]:
        raise HTTPException(status_code=503, detail="Providers not ready")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.pending,
        "captions": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    import asyncio
    asyncio.create_task(_run_job(job_id, req.video_url, req.styles))

    return ProcessAccepted(job_id=job_id, status=JobStatus.pending)


# ---------- Poll status ----------

@app.get("/status/{job_id}", response_model=StatusResponse, dependencies=[Depends(require_api_key)])
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        captions=job["captions"],
        error=job["error"],
    )


# ---------- Background worker ----------

async def _run_job(job_id: str, video_url: str, styles: list[str]) -> None:
    try:
        # --- Step 1: download ---
        try:
            video_path = await download_video(job_id, video_url)
        except Exception as exc:
            logger.error("Download failed for %s: %s", job_id, exc)
            jobs[job_id]["status"] = JobStatus.error
            jobs[job_id]["error"] = f"Download failed: {exc}"
            return

        # --- Step 2: pick a description provider ---
        desc_providers = state["desc_providers"]
        provider = desc_providers[hash(job_id) % len(desc_providers)]

        video_ref = None
        keyframe_paths = None
        try:
            video_ref = await provider.upload_video(video_path)
        except (UploadError, Exception) as exc:
            logger.warning("Upload failed for %s, falling back to keyframes: %s", job_id, exc)
            try:
                keyframe_paths = await extract_keyframes(video_path, job_id)
            except Exception as kf_exc:
                jobs[job_id]["status"] = JobStatus.error
                jobs[job_id]["error"] = f"Keyframe extraction failed: {kf_exc}"
                return

        # --- Step 3: description ---
        try:
            description = await provider.generate_description(video_ref, keyframe_paths)
        except (GenerationError, Exception) as exc:
            jobs[job_id]["status"] = JobStatus.error
            jobs[job_id]["error"] = f"Description generation failed: {exc}"
            return

        # --- Step 4: caption per style ---
        caption_providers = state["caption_providers"]
        caption_provider, _cooldown = caption_providers[hash(job_id) % len(caption_providers)]

        captions: dict[str, str] = {}
        for style in styles:
            try:
                caption = await caption_provider.generate_caption(description, style)
                captions[style] = sanitize_caption(caption)
            except (GenerationError, Exception) as exc:
                logger.warning("Caption failed for %s style %s: %s", job_id, style, exc)
                captions[style] = FALLBACK_CAPTION

        jobs[job_id]["status"] = JobStatus.done
        jobs[job_id]["captions"] = captions

    except Exception as exc:  # catch-all so a bug never leaves a job stuck at "pending" forever
        logger.exception("Unexpected error processing job %s", job_id)
        jobs[job_id]["status"] = JobStatus.error
        jobs[job_id]["error"] = f"Unexpected error: {exc}"