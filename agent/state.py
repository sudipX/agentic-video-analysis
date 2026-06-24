from typing import TypedDict, Optional

class VideoAnalysisState(TypedDict):

    #input
    video_path : str
    video_filename : str

    #Populated by frame extractor
    frames : list

    #Populated by motion_analyzer
    frames_with_motion : list
    any_motion_found : bool

    #Populated by object_detector
    detections_by_frame : dict

    #Populated by summarizer
    summary_text : str
    key_moments : list

    #Populatd by report_writer
    final_report : dict

    #Retry/error bookkeeping
    current_step : str
    retry_count : int
    error_message : Optional[str]
    failed : bool