"""
tracking.py
------------
Handles multi-object tracking using DeepSort
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Tuple

class Tracker:
    def __init__(self, max_age: int=90, max_cosine_distance: float=0.3, n_init: int=3):
        """
        Initiates tracking using DeepSort tracker.

        Args:
            max_age(int): Maximum number of frames to keep lost tracks alive.
            max_cosine_distance(float): Threshold of object's similarity.
            n_init(int): Number of consecutive detections before track is confirmed.
        """
        self.tracker = DeepSort(max_age=max_age, max_cosine_distance=max_cosine_distance, n_init=n_init)

    def update_tracker(self, detections: List, frame: np.ndarray) -> List[Tuple[List[int], float, int]]:
        """
        Updates tracker with new detections.

        Args:
            detections(List): List of detected bounding boxes with format ([x,y,w,h], conf, cls)
            frame(np.ndarray): Current video frames.

        Returns:
            List: Active tracks after update. Each track has:
                    - track_id(int)
                    - bbox(list[int]): [x1, y1, x2, y2]
                    - class_id(int)
                    - confidence(float)
        """
        return self.tracker.update_tracks(detections, frame=frame)