from langgraph.graph import StateGraph, END
from agent.state import VideoAnalysisState
from agent.error_handling import route_after_error_check
from agent.nodes import (
    frame_extractor_node,
    motion_analyzer_node,
    object_detector_node,
    summarizer_node,
    report_writer_node,
    route_after_motion_analysis
)


def build_video_analysis_graph():
    builder = StateGraph(VideoAnalysisState)

    builder.add_node("frame_extractor", frame_extractor_node)
    builder.add_node("motion_analyzer", motion_analyzer_node)
    builder.add_node("object_detector", object_detector_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("report_writer", report_writer_node)

    builder.set_entry_point("frame_extractor")

    builder.add_conditional_edges(
        "frame_extractor",
        route_after_error_check,
        {
            "continue": "motion_analyzer",
            "retry": "frame_extractor",
            "give_up": "report_writer"
        }
    )

    builder.add_conditional_edges(
        "motion_analyzer",
        route_after_motion_analysis,
        {
            "run_object_detection": "object_detector",
            "skip_to_summary": "summarizer"
        }
    )

    builder.add_conditional_edges(
        "object_detector",
        route_after_error_check,
        {
            "continue": "summarizer",
            "retry": "object_detector",
            "give_up": "report_writer"
        }
    )

    builder.add_edge("summarizer", "report_writer")
    builder.add_edge("report_writer", END)

    return builder.compile()


# Module-level compiled graph, imported by main.py
video_analysis_app = build_video_analysis_graph()


if __name__ == "__main__":
    initial_state = {
        "video_path": "sample_video.mp4",
        "video_filename": "sample_video.mp4",
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
    print("\nFINAL REPORT")
    print(final_state["final_report"])