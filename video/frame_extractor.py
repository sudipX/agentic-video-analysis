import cv2
import os
from pathlib import Path

def extract_frames (video_path : str, output_dir : str, sample_fps : float = 1.0) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video file : {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    if native_fps <= 0.0: # edge case for some malformed video files
        native_fps = 30.0
        print("Couldn't read native FPS. Assuming 30 fps")

    frame_interval = max(1, round(native_fps/sample_fps))
    print(f"Native fps : {native_fps}. sampling every {frame_interval}")

    video_stem = Path(video_path).stem
    extracted = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_count%frame_interval==0:
            timestamp_sec = frame_count / native_fps

            file_name = f"{video_stem}_frame{saved_count:04d}.jpg"
            saved_path = output_path/file_name

            cv2.imwrite(str(saved_path),frame)

            extracted.append({
                "frame_path" : str(saved_path),
                "timestamp_sec" : round(timestamp_sec,2),
                "frame_index" : saved_count 
            })

            saved_count+=1

        frame_count+=1

    cap.release()

    print(f"Extracted {saved_count} frames from {frame_count} total frames.")

    return extracted



          

