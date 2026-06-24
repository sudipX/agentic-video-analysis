import cv2
import numpy as np

def compute_motion_score(frame_path_a : str, frame_path_b : str) ->float:


    frame_a = cv2.imread(frame_path_a, cv2.IMREAD_GRAYSCALE)
    frame_b = cv2.imread(frame_path_b, cv2.IMREAD_GRAYSCALE)

    if frame_a is None or frame_b is None:
        raise ValueError(f"Could not find one of {frame_path_a} or {frame_path_b}")
    

    #Here (21,21) is the size of gaussain blur kernal
    frame_a_blurred = cv2.GaussianBlur(frame_a,(21,21),0)
    frame_b_blurred = cv2.GaussianBlur(frame_b,(21,21),0)

    diff = cv2.absdiff(frame_a_blurred,frame_b_blurred)

    # Here is any pixel difference greater than 25 is treated as changed. Then converted into binary image, where changed is 1, else 0.
    _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    changed_pixel = np.count_nonzero(thresholded)
    total_pixel = thresholded.size
    motion_score = (changed_pixel/total_pixel)*100

    return motion_score

def detect_motion_in_sequence(frame_records : list, motion_threshold : float = 2.0) -> list:

    results = []

    for i, record in enumerate(frame_records):

        if i==0:
            record = {**record,"motion_score":0.0,"has_motion":False}

        else:
            previous_frame = frame_records[i-1]["frame_path"]
            current_frame = record["frame_path"]

            score = compute_motion_score(previous_frame,current_frame)

            has_motion = score>=motion_threshold

            record = {**record, "motion_score":score,"has_motion" : has_motion}
        
        results.append(record)

    motion_count = sum(1 for r in results if r["has_motion"])
    print(f"Motion detected in {motion_count} frames out of {len(results)} frames.")

    return results
