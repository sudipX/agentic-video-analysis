from ultralytics import YOLO
import cv2

MODEL_NAME = "yolov8n.pt"

yolo_model = YOLO(MODEL_NAME)

CONFIDENCE_THRESHOLD = 0.5

def detect_objects (image_path : str) -> list:

    # verbose suppresses YOLO's default per-image console logging.
    results = yolo_model(image_path,verbose=False)

    # Taking the one and only image, a result object
    result = results[0]

    detections = []

    print(f"[DEBUG] {image_path} — raw detections: {len(result.boxes)}")
    for box in result.boxes:
        print(f"  class={result.names[int(box.cls.item())]} conf={float(box.conf.item()):.3f}")


    # result.boxes gives all the bounding boxes for this image.
    for box in result.boxes:

        confidence = float(box.conf.item()) #box.conf is a tensor containing the confidence score

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls.item()) #box.cls is a tensor containing the numerics class id

        class_name = result.names[class_id] # result.name maps the class id to the human_readable names
        
        bbox = box.xyxy[0].tolist() # converting a (1,4) tensor into flat python list
        bbox = [round(coord,1) for coord in bbox]

        detections.append({
            "class_name" : class_name,
            "confidence" : round(confidence,3),
            "bbox" : bbox
        })
    
    return detections

def detect_objects_batch(image_paths:list) -> dict :
    
    results = {}

    for path in image_paths:
        try:
            detections = detect_objects(path)
            results[path] = detections
            print(f"{path} : {len(detections)} objects detected.")

        except Exception as e:
            print(f"Skipped {path} due to error : {e}")
            results[path] = []
    
    return results

# if __name__ == '__main__':
#     detections = detect_objects("saved_image.jpeg")
#     for d in detections:
#         print(f"{d["class_name"]} ({d["confidence"]}) at {d["bbox"]}")
