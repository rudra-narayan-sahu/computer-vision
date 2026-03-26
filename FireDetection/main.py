import cv2
import argparse
from ultralytics import YOLO
import os

def run_detection(source, model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check if source is a camera index
    if source.isdigit():
        source = int(source)
    
    # Open the source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        # If it's an image, just load it
        img = cv2.imread(source)
        if img is not None:
            print(f"Detecting on image: {source}")
            results = model(img)
            results[0].show()
        else:
            print(f"Error: Could not open source {source}")
        return

    print(f"Starting detection on source: {source} (Press 'q' to quit)")
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # results = model.predict(frame, conf=0.5) # Removed prediction to match old behavior if needed, but let's keep it robust
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Fire Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection using YOLOv8")
    parser.add_argument("--source", type=str, default="fireIMg.jpg", help="Source (image, video, or camera index)")
    parser.add_argument("--model", type=str, default="yolov8n_fire_smoke.pt", help="Path to YOLOv8 model")
    args = parser.parse_args()
    
    run_detection(args.source, args.model)