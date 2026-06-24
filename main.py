import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent.graph import video_analysis_app

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubmitVideoResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class KeyMoment(BaseModel):
    timestamp_sec: float
    description: str


class AnalysisReport(BaseModel):
    video_filename: str
    total_frames_sampled: int
    frames_with_motion: int
    total_objects_detected: int
    summary: str
    key_moments: list[KeyMoment]
    analysis_failed: bool


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    report: Optional[AnalysisReport] = None
    error: Optional[str] = None

jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Agentic Video Analysis API")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Agentic Video Analysis API",
    description="Submit a video for analysis. Poll for the structured report.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_analysis_job(job_id: str, video_path: str, video_filename: str):
    
    # Runs the full LangGraph pipeline in the background after the /analyze endpoint has already returned HTTP 202 to the client. Updates the jobs dict with the result or error when finished.
    
    jobs[job_id]["status"] = JobStatus.PROCESSING

    try:
        initial_state = {
            "video_path": video_path,
            "video_filename": video_filename,
            "frames": [],
            "frames_with_motion": [],
            "any_motion_found": False,
            "detections_by_frame": {},
            "summary_text": "",
            "key_moments": [],
            "final_report": {},
            "current_step": "",
            "retry_count": 0,
            "error_message": None,
            "failed": False
        }

        final_state = video_analysis_app.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["report"] = final_state["final_report"]

    except Exception as e:
        print(f"[run_analysis_job] Job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"]  = str(e)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Agentic Video Analysis API is running.",
        "docs": "Visit /docs for interactive API documentation."
    }


@app.get("/jobs/count")
async def job_count():
    #Returns how many jobs are currently tracked in memory.
    return {
        "total": len(jobs),
        "pending": sum(1 for j in jobs.values() if j["status"] == JobStatus.PENDING),
        "processing": sum(1 for j in jobs.values() if j["status"] == JobStatus.PROCESSING),
        "completed": sum(1 for j in jobs.values() if j["status"] == JobStatus.COMPLETED),
        "failed": sum(1 for j in jobs.values() if j["status"] == JobStatus.FAILED),
    }


@app.post(
    "/analyze",
    response_model=SubmitVideoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video for analysis"
)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A video file to analyze")
):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: '{file_ext}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded video to disk
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        with open(save_path, "wb") as dest:
            shutil.copyfileobj(file.file, dest)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded video: {str(e)}"
        )

    # Create job record BEFORE scheduling background task
    # so any immediate poll returns a valid record, not a 404
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "report": None,
        "error":  None
    }

    # Schedule the analysis to run after this response is returned
    background_tasks.add_task(
        run_analysis_job,
        job_id=job_id,
        video_path=str(save_path),
        video_filename=file.filename
    )

    return SubmitVideoResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video accepted. Analysis running in the background. "
                f"Poll GET /jobs/{job_id} to check progress."
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Check the status of an analysis job"
)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No job found with id: {job_id}"
        )

    job = jobs[job_id]

    report_model = None
    if job["report"] is not None:
        # key_moments from Gemini is a list of plain dicts.
        # Pydantic needs them as KeyMoment objects.
        key_moments = [
            KeyMoment(**km)
            for km in job["report"].get("key_moments", [])
            if isinstance(km, dict)
                and "timestamp_sec" in km
                and "description" in km
        ]
        report_model = AnalysisReport(
            video_filename=         job["report"].get("video_filename", ""),
            total_frames_sampled=   job["report"].get("total_frames_sampled", 0),
            frames_with_motion=     job["report"].get("frames_with_motion", 0),
            total_objects_detected= job["report"].get("total_objects_detected", 0),
            summary=                job["report"].get("summary", ""),
            key_moments=            key_moments,
            analysis_failed=        job["report"].get("analysis_failed", False)
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        report=report_model,
        error=job["error"]
    )