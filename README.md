# Agentic Video Analysis System

A stateful agentic pipeline that processes video through conditional computer vision steps and produces a structured natural-language report of what happened — built with LangGraph, OpenCV, and YOLOv8, running fully locally via Ollama.

---

## What It Does

Given a video file, the system:

1. Samples frames at a fixed rate (1 fps by default)
2. Detects motion between consecutive frames using frame differencing
3. **Only runs object detection on frames where motion was found** — skipping expensive YOLO inference on static segments entirely
4. Identifies objects in motion frames (person, car, bicycle, dog — 80 COCO classes)
5. Converts the structured detections into a natural-language summary and a timestamped JSON list of key moments

The conditional routing — run YOLO or skip it based on what the motion detector found — is a real agentic decision made at runtime by the LangGraph state machine, not a hardcoded if/else in a script.

Tested on real video footage: the system correctly identified motion timestamps, detected objects per frame, and produced a coherent narrative summary without human intervention.

---

## Architecture

```
Video Input
    │
    ▼
┌─────────────────┐
│ Frame Extractor │  OpenCV — sample 1 frame/sec, save to disk
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Motion Analyzer  │  Frame differencing + Gaussian blur
└────────┬─────────┘
         │
    Motion found?
    ┌────┴────┐
   YES        NO
    │          │
    ▼          │
┌──────────────────┐  │
│ Object Detector  │  │  YOLOv8n — 80-class detection
│    (YOLO)        │  │  Skipped entirely if no motion
└────────┬─────────┘  │
         └────┬────────┘
              │
              ▼
┌──────────────────┐
│   Summarizer     │  Ollama (llama3.2:3b) — structured
│                  │  CV output → natural language
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Report Writer   │  Assembles final JSON report
└──────────────────┘
```

**Each box is a LangGraph node. Each arrow is an edge. The motion branch is a conditional edge decided at runtime.**

Any node can fail and retry up to 3 times before the pipeline gracefully produces a partial report — implemented as a decorator wrapping risky nodes, not manual try/except at every call site.

---

## Why This Architecture Matters

Most "agentic AI" projects chain a few LLM calls in sequence. This one is different:

- **Real computer vision underneath.** Motion detection uses frame differencing with Gaussian blur (a deliberate choice over optical flow — faster as a cheap gating step before expensive YOLO inference). Object detection uses YOLOv8, which processes entire frames in one forward pass rather than two-stage region-proposal approaches.

- **The LLM never sees raw video.** It receives a structured, timestamped timeline of events (e.g., `"t=4.0s: motion detected, objects seen: 1 person, 1 car"`) built entirely by OpenCV and YOLO. The LLM's job is synthesis — turning structured data into readable prose — not perception.

- **The conditional skip is real efficiency.** In a 60-second video at 1fps, if 40 frames are static, YOLO never runs on those 40 frames. The motion detector handles them cheaply in milliseconds each.

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent Orchestration | LangGraph |
| Web Framework | FastAPI |
| Frame Extraction | OpenCV |
| Motion Detection | OpenCV (frame differencing) |
| Object Detection | YOLOv8n (Ultralytics) |
| LLM | Ollama (llama3.2:3b, runs locally) |
| Validation | Pydantic v2 |

**No LangChain. No paid APIs. No cloud dependencies. Runs entirely locally.**

---

## Project Structure

```
agentic-video-analysis/
├── main.py                      # FastAPI: routes, background tasks, job store
├── models.py                    # Pydantic schemas including JobStatus enum
├── requirements.txt
│
├── video/
│   ├── frame_extractor.py       # OpenCV: sample frames from video file
│   ├── motion_detector.py       # Frame differencing: motion score per frame pair
│   └── object_detector.py       # YOLOv8: per-frame object detection
│
├── llm/
│   └── summarizer.py            # Ollama: timeline → natural language + key moments JSON
│
├── agent/
│   ├── state.py                 # VideoAnalysisState TypedDict — shared graph state
│   ├── nodes.py                 # All node functions + conditional routing function
│   ├── error_handling.py        # Retry decorator + route_after_error_check
│   └── graph.py                 # StateGraph assembly, compilation, module-level app
│
├── uploads/                     # Uploaded videos (auto-created)
└── extracted_frames/            # Extracted frame images (auto-created)
```

---

## Setup and Installation

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/agentic-video-analysis.git
cd agentic-video-analysis
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull the Ollama model**
```bash
ollama pull llama3.2:3b
```

**5. Start the server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

> Runs on port **8001** to avoid conflict with the RAG pipeline on 8000 if both are running simultaneously.

---

## Using the API

This API uses a **submit-and-poll** pattern. Video analysis can take minutes — the server accepts your video immediately and returns a job ID. You poll separately to retrieve the result.

### Option A — Interactive UI

Open your browser at `http://localhost:8001/docs`

### Option B — curl

**Step 1: Submit a video**
```bash
curl -X POST http://localhost:8001/analyze \
     -F "file=@/path/to/your/video.mp4"
```

**Response (immediate, under 1 second):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "message": "Video accepted. Analysis running in the background. Poll GET /jobs/a1b2c3d4-... to check progress."
}
```

**Step 2: Poll for the result**
```bash
curl http://localhost:8001/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**While processing:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "processing",
  "report": null,
  "error": null
}
```

**When complete:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "report": {
    "video_filename": "traffic_clip.mp4",
    "total_frames_sampled": 30,
    "frames_with_motion": 12,
    "total_objects_detected": 18,
    "summary": "The video shows a quiet street scene for the first 8 seconds. At approximately 9 seconds, a person enters from the left side of frame and walks across. A car passes through the background at around 14 seconds. The remainder of the clip is static.",
    "key_moments": [
      {"timestamp_sec": 9.0, "description": "A person enters from the left side of frame"},
      {"timestamp_sec": 14.0, "description": "A car passes through the background"}
    ],
    "analysis_failed": false
  },
  "error": null
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/jobs/count` | Live count of jobs by status |
| `POST` | `/analyze` | Submit a video for analysis (returns immediately) |
| `GET` | `/jobs/{job_id}` | Poll for job status and result |

**Supported video formats:** `.mp4`, `.avi`, `.mov`, `.mkv`

---

## LangGraph State

Every node in the graph reads from and writes to a single shared state object:

```python
class VideoAnalysisState(TypedDict):
    video_path:          str
    video_filename:      str
    frames:              list        # populated by frame_extractor_node
    frames_with_motion:  list        # populated by motion_analyzer_node
    any_motion_found:    bool        # used by conditional routing function
    detections_by_frame: dict        # populated by object_detector_node
    summary_text:        str         # populated by summarizer_node
    key_moments:         list        # populated by summarizer_node
    final_report:        dict        # populated by report_writer_node
    current_step:        str         # bookkeeping for retry logic
    retry_count:         int         # incremented by error handler
    error_message:       Optional[str]
    failed:              bool
```

The conditional edge after `motion_analyzer_node` reads `any_motion_found` and routes to either `object_detector_node` or directly to `summarizer_node`. This is a plain Python function — no LLM call, no magic, just a dict lookup that returns a string key.

---

## Error Handling

Risky nodes (`frame_extractor_node`, `object_detector_node`) are wrapped with a decorator:

```python
@with_error_handling("frame_extractor")
def frame_extractor_node(state: dict) -> dict:
    ...
```

If a node raises an exception, instead of crashing the graph, the decorator catches it, writes the error to `state["error_message"]`, and increments `state["retry_count"]`. The graph then routes to a check function that decides: retry the node, or give up and route to `report_writer` with `failed=True`.

This means even failed analyses produce a structured response — a partial report with `"analysis_failed": true` — rather than a hung job or an unhandled 500.

---

## Design Decisions

**Why motion gating before YOLO?** YOLOv8 inference costs real compute per frame. Frame differencing with Gaussian blur costs almost nothing. For a 60-second video at 30fps with 20 seconds of actual activity, motion gating means YOLO only processes ~20 frames instead of 1800. The motion threshold (2% of pixels changed) is tunable and documented as a deliberate tradeoff — lower catches more activity, higher is more noise-resistant.

**Why frame differencing instead of optical flow?** Optical flow (including RAFT, which I implemented from scratch separately) gives richer per-pixel motion vectors — direction, magnitude — but is an order of magnitude slower. For a binary "did something move?" gate before YOLO, frame differencing provides the necessary signal at a fraction of the compute cost. Optical flow would be the right swap if the use case shifted to requiring motion direction or speed (traffic speed estimation, for example).

**Why LangGraph instead of a plain script?** A script has control flow baked into source code. LangGraph has control flow as data — nodes, edges, and routing functions that can be inspected, tested, and modified independently. The conditional skip, the retry logic, and the error recovery would each require nested if/else and manual state threading in a plain script. In LangGraph they are each one declarative `add_conditional_edges` call.

**Why the submit-and-poll pattern?** A synchronous `/analyze` endpoint that blocks until the LangGraph agent finishes would time out for any video longer than ~30 seconds — most web servers and load balancers enforce 30–60 second request timeouts. `BackgroundTasks` + job polling decouples the HTTP layer from processing time entirely.

---

## Real-World Applications

This architecture — cheap motion filtering as a first-pass gate before expensive object detection — is the core pattern behind:

- **Post-event security footage review:** produce a timestamped narrative of what happened without watching every minute of footage
- **Wildlife camera trap analysis:** filter hours of empty footage before running detection on frames with actual animals
- **Sports and training video review:** automatically identify periods of activity vs. static breaks
- **Manufacturing pipeline monitoring:** detect when something moved that shouldn't have

Extending to live-stream processing would require replacing the batch video file input with a streaming frame source — the per-frame logic (motion → YOLO → summarize) carries over directly.

---

## Known Limitations

- **In-memory job store:** job records live in the server process's RAM. Restarting the server loses all job history. For production, replace the `jobs` dict with Redis or a database-backed store.
- **Single process:** the in-memory job store does not work correctly across multiple server replicas or Uvicorn workers. Suitable for single-process deployment only.
- **No live stream support:** the system processes uploaded video files. Continuous live-stream processing would require a different input architecture.
- **LLM hallucination risk on sparse timelines:** if very little motion is detected, the LLM has little to work with. The prompt explicitly instructs it not to speculate beyond the timeline, but sparse input produces sparse output.

---

## Requirements

```
fastapi
uvicorn[standard]
python-multipart
python-dotenv
langgraph
opencv-python
ultralytics
ollama
numpy
pydantic
```