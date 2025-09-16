"""
detections.py
-------------
Handles object detection using YOLOv8.
"""

from ultralytics import YOLO
import numpy as np
import logging
from typing import List, Tuple

class Detector:
    def __init__(self, model_path: str, imgsz: int = 640):
        """
        Initiates YOLOv8 detection

        Args:
            model_path(str): Path to YOLO weights.
            imgsz(int): Image size (default=640)
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.imgsz = imgsz
        self.device = self.model.device

        logging.info(f"YOLOv8 initialized on {self.device} with image size {self.imgsz}")

    def detect(self, frame: np.ndarray, target_classes: List[int], conf_thresh: float = 0.4) -> List[Tuple[List[int], float, int]]:
        """
        Run YOLOv8 on frames and return detections (([x1, y1, w, h], conf, cls))

        Args:
            frame(np.ndarray): Input video frames.
            target_classes(list[int]): Class IDs to detect object (person, car, motorcycle, etc.).
            conf_thresh(float): Confidence threshold.

        Returns:
            list: A list of detections in format:
                    [([x,y,w,h], confidence, class_id)]
                    where (x, y) is top-left corner of the bounding box.
        """
        if not target_classes:
            raise ValueError("target_classes cannot be empty")
        
        target_set = set(target_classes)
        detections = []

        # Run model
        results = self.model.predict(frame, imgsz=self.imgsz, conf=conf_thresh, verbose=False, stream=True)

        # Process result
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int,box.xyxy[0])
                w, h = x2-x1, y2-y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Keep only target classes with sufficient threshold
                if cls in target_set and conf >= conf_thresh:
                    detections.append(([x1, y1, w, h], conf, cls))

        return detections