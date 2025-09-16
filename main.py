"""
main.py
--------
Main script for running the Vehicle & Pedestrian Tracking pipeline.

Pipeline:
1. Load YOLOv8 for object detection.
2. Track objects using DeepSort.
3. Count unique objects and display FPS.
"""
import os
import cv2
import argparse
import logging
from detections import Detector
from tracking import Tracker
from utils import Visualizer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
class_ids = [0,1,2,3,5,7]  # person, bicycle, car, motorcycle, bus, truck

def parse_args():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser(description="Vehicle & Pedestrian Tracking with YOLOv8 + DeepSORT")
    parser.add_argument("--input", type=str, default="data/crosswalk_traffic.mp4", help="Path to input video or 0 for webcam")
    parser.add_argument("--output", type=str, default="output/crosswalk_traffic_output.mp4", help="Path to save the output")
    parser.add_argument("--model", type=str, default="Yolo-Weights/yolov8m.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--frame_width", type=int, default=640, help="Resize display width (0 = no resize)")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold for displaying & counting detections")
    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Ensure input directory exists (for user convenience)
    input_dir = os.path.dirname(args.input)
    if input_dir and not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        logging.warning(f"Input directory {input_dir} did not exist. Created it. "
                        f"Please upload your test videos there.")

    # Check if input file exists (unless webcam)
    if args.input != "0" and not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    # Initialize detector, tracker and visualizer
    logging.info("Initializing YOLOv8 detector and DeepSORT tracker...")
    detector = Detector(args.model)
    tracker = Tracker()
    visualizer = Visualizer(class_ids, detector.class_names)

    # Open video (or webcam if input == 0)
    cap = cv2.VideoCapture(0 if args.input=="0" else args.input)
    if not cap.isOpened():
        logging.error(f"Cannot open {args.input}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps is None:
        fps = 30.0

    if not args.output.lower().endswith(".mp4"):
        args.output = os.path.splitext(args.output)[0] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # good cross-platform choice
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame, class_ids, args.conf_thresh)

        # Track Objects
        tracks = tracker.update_tracker(detections, frame)

        # Draw bounding box and add label
        visualizer.draw_tracks(frame, tracks, conf_thresh=args.conf_thresh)
        visualizer.draw_counts(frame)
        visualizer.draw_fps(frame, show_fps=False)

        # Save results
        out.write(frame)

        # Resize for display if needed 
        if args.frame_width > 0: 
            scale = args.frame_width/width 
            resized_frame = cv2.resize(frame, (args.frame_width, int(height * scale))) 
        else: 
            resized_frame = frame

        # Show results
        cv2.imshow("Vehicle & Pedestrian Tracking", resized_frame)

        # Quit Key
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Processing finished. Output saved at {args.output}.")

if __name__ == "__main__":
    main()