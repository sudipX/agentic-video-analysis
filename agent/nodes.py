# file: agent/nodes.py

from pathlib import Path
from agent.error_handling import with_error_handling
from video.frame_extractor import extract_frames
from video.motion_detector import detect_motion_in_sequence
from video.object_detector import detect_objects_batch
from llm.summarizer import summarize_video, extract_key_moments

FRAME_OUTPUT_DIR = "./extracted_frames"


@with_error_handling("frame_extractor")
def frame_extractor_node(state: dict) -> dict:
    print(f"[frame_extractor_node] Processing: {state['video_filename']}")
    frames = extract_frames(
        video_path=state["video_path"],
        output_dir=FRAME_OUTPUT_DIR,
        sample_fps=1.0
    )
    return {
        "frames":        frames,
        "current_step":  "frame_extractor",
        "retry_count":   0,
        "error_message": None
    }


def motion_analyzer_node(state: dict) -> dict:
    print("[motion_analyzer_node] Analyzing motion between frames")
    frames_with_motion = detect_motion_in_sequence(
        frame_records=state["frames"],
        motion_threshold=2.0
    )
    any_motion_found = any(f["has_motion"] for f in frames_with_motion)
    return {
        "frames_with_motion": frames_with_motion,
        "any_motion_found":   any_motion_found,
        "current_step":       "motion_analyzer",
        "retry_count":        0,
        "error_message":      None
    }


@with_error_handling("object_detector")
def object_detector_node(state: dict) -> dict:
    print("[object_detector_node] Running YOLO on frames with motion")
    motion_frame_paths = [
        f["frame_path"]
        for f in state["frames_with_motion"]
        if f["has_motion"]
    ]
    detections_by_frame = detect_objects_batch(motion_frame_paths)

    print(f"[DEBUG] object_detector RETURNING: {len(detections_by_frame)} frames with detections")
    print(f"[DEBUG] sample: {list(detections_by_frame.items())[:1]}")

    
    return {
        "detections_by_frame": detections_by_frame,
        "current_step":        "object_detector",
        "retry_count":         0,
        "error_message":       None
    }


def summarizer_node(state: dict) -> dict:

    print(f"[DEBUG] summarizer received state keys: {list(state.keys())}")
    print(f"[DEBUG] summarizer detections_by_frame: {len(state.get('detections_by_frame', {}))}")


    print("[summarizer_node] Calling Gemini for summary and key moments")
    detections_by_frame = state.get("detections_by_frame", {})

    # DEBUG
    print(f"[DEBUG] detections_by_frame keys: {list(detections_by_frame.keys())[:2]}")
    print(f"[DEBUG] frames_with_motion paths: {[f['frame_path'] for f in state['frames_with_motion'][:2]]}")


    summary_text = summarize_video(
        video_filename=state["video_filename"],
        frame_records=state["frames_with_motion"],
        detections_by_frame=detections_by_frame
    )
    key_moments = extract_key_moments(
        video_filename=state["video_filename"],
        frame_records=state["frames_with_motion"],
        detections_by_frame=detections_by_frame
    )
    return {
        "summary_text":  summary_text,
        "key_moments":   key_moments,
        "current_step":  "summarizer",
        "retry_count":   0,
        "error_message": None
    }


def report_writer_node(state: dict) -> dict:
    print("[report_writer_node] Assembling final report")
    total_frames     = len(state.get("frames", []))
    motion_frames    = sum(
        1 for f in state.get("frames_with_motion", []) if f.get("has_motion")
    )
    detections       = state.get("detections_by_frame", {})
    total_detections = sum(len(d) for d in detections.values())

    report = {
        "video_filename":          state["video_filename"],
        "total_frames_sampled":    total_frames,
        "frames_with_motion":      motion_frames,
        "total_objects_detected":  total_detections,
        "summary":                 state.get("summary_text", ""),
        "key_moments":             state.get("key_moments", []),
        "analysis_failed":         state.get("failed", False)
    }
    return {
        "final_report": report,
        "current_step": "report_writer"
    }


def route_after_motion_analysis(state: dict) -> str:
    if state.get("any_motion_found", False):
        print("[route] Motion found -> running object detector")
        return "run_object_detection"
    else:
        print("[route] No motion found -> skipping to summary")
        return "skip_to_summary"