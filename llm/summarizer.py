import json
import re
from ollama import Client

client = Client()
MODEL = "llama3.2:3b"


def _call_llm(prompt: str) -> str:
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"]


def _strip_markdown_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_summary_prompt(
    video_filename: str,
    frame_records: list,
    detections_by_frame: dict,
) -> str:
    timeline_lines = []

    for record in frame_records:
        if not record.get("has_motion", False):
            continue

        timestamp = record["timestamp_sec"]
        path = record["frame_path"]
        detections = detections_by_frame.get(path, [])

        if detections:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
            objects_str = ", ".join(f"{v} {k}" for k, v in counts.items())
            timeline_lines.append(
                f"t={timestamp:.1f}s: motion detected, objects seen: {objects_str}"
            )
        else:
            timeline_lines.append(
                f"t={timestamp:.1f}s: motion detected, no recognisable objects"
            )

    timeline_text = (
        "\n".join(timeline_lines)
        if timeline_lines
        else "No motion detected throughout the video."
    )

    return f"""You are a video analysis assistant. Below is a timeline of automatically detected events from a video file, produced by a motion detector and a YOLO object detection model. Write a clear, natural-language summary of what likely happened in the video, based only on this timeline.

INSTRUCTIONS:
- Describe the sequence of events in plain language, as if narrating the video to someone who cannot watch it.
- Mention approximate timestamps for key events.
- If the timeline is sparse or empty, say so honestly rather than inventing detail.
- Do not speculate about anything not supported by the timeline data.
- Keep the summary to 3-5 sentences.

VIDEO FILE: {video_filename}

DETECTED EVENT TIMELINE:
{timeline_text}

YOUR SUMMARY:"""


_JSON_INSTRUCTION = """

Now, instead of prose, output ONLY a JSON array of the most notable moments,
with no preamble, no markdown code fences, and no explanation. Use exactly
this shape:
[
  {"timestamp_sec": 1.0, "description": "A person enters the frame"},
  {"timestamp_sec": 4.5, "description": "A car passes through"}
]
If there are no notable moments, output an empty array: []"""


def summarize_video(
    video_filename: str,
    frame_records: list,
    detections_by_frame: dict,
    max_tokens: int = 512,
) -> str:
    prompt = build_summary_prompt(video_filename, frame_records, detections_by_frame)
    return _call_llm(prompt)


def extract_key_moments(
    video_filename: str,
    frame_records: list,
    detections_by_frame: dict,
    max_tokens: int = 512,
) -> list:
    prompt = build_summary_prompt(video_filename, frame_records, detections_by_frame)
    raw = _call_llm(prompt + _JSON_INSTRUCTION)
    raw = _strip_markdown_fences(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse key moments JSON: {e}")
        print(f"Raw response was: {raw[:200]}")
        return []