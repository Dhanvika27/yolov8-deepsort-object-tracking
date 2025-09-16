"""
utils.py
---------
Utility functions for drawing bounding boxes, counting objects, and calculating FPS.
"""

import cv2
import time
import numpy as np
import random
from typing import List, Dict, Tuple

def generate_colors(class_ids: List[int]) -> Dict[int, Tuple[int,int,int]]:
        """
        Generate random colors for each class ID.
        
        Args:
            class_ids (list[int]): List of class IDs.
        Returns:
            dict: {class_id: (B, G, R)}
        """
        colors= {}
        random.seed(42)  # reproducibility
        for cid in class_ids:
            colors[cid] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return colors
        

class Visualizer:
    def __init__(self, class_ids: List[int], class_names: Dict[int, str]):
        """
        Initializes the visualizer with tracking and counting utilities.

        Args:
            class_ids (list[int]): Target class IDs.
            class_names (dict): Mapping from class IDs to class names.
        """
        self.class_ids = class_ids
        self.class_names = class_names
        self.colors = generate_colors(class_ids)
        self.unique_ids = set()
        self.counts = {cid: 0 for cid in class_ids}
        self.prev_time = time.time()
        self.fps_history = []

    def draw_tracks(self, frame: np.ndarray, tracks: list, conf_thresh: float=0.4):
        """
        Draw bounding box on tracked objects and update counts.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            cls = track.get_det_class() if track.get_det_class() is not None else -1
            if cls == -1 or cls not in self.class_ids:
                continue

            conf = getattr(track, "det_conf", None)
            if conf is None or conf < conf_thresh:
                continue

            label = f"ID: {track_id} {self.class_names[cls]} {conf*100:.1f}%"
            color = self.colors.get(cls, (255,255,255))

            # Draw bounding box and add label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_y = max(y1 - 5, th + 5)
            cv2.rectangle(frame, (x1, text_y - th - 3), (x1 + tw + 5, text_y + 2), color, cv2.FILLED)

            # Add label text
            cv2.putText(frame, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

            # Update counts for unique IDs
            if (track_id, cls) not in self.unique_ids:
                self.unique_ids.add((track_id, cls))
                self.counts[cls] += 1

    def draw_counts(self, frame: np.ndarray):
        """
        Display total counts for each selected class.

        Args:
            frame(np.ndarray): Video frames.
            class_names(dict): Mapping of class ids to names.
            class_ids(list[int]): List of target class ids.
        """
        x_offset = 20
        y_offset = 20

        for cid in self.class_ids:
            text = f"{self.class_names[cid]}: {self.counts[cid]}"
            
            # Get text size for background box
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw filled rectangle background
            cv2.rectangle(frame, (x_offset - 5, y_offset - th - 5), (x_offset + tw + 5, y_offset + 5), (0,0,0), cv2.FILLED)

            # Add text on top of rectangle
            cv2.putText(frame, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            y_offset += th + 15

    def draw_fps(self, frame: np.ndarray, show_fps=False):
        """
        Display FPS on frame.

        Args:
            frame(np.ndarray): Video frames.
        """
        curr_time = time.time()
        fps = 1/ (curr_time - self.prev_time)
        self.prev_time = curr_time

        self.fps_history.append(fps)
        if len(self.fps_history)> 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history)/ len(self.fps_history)
        
        if show_fps:
            text = f"FPS: {avg_fps:.2f}"
            cv2.putText(frame, text, (frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    def reset_counts(self):
        """
        Reset counts and IDs for new video.
        """
        self.unique_ids.clear()
        self.counts = {cid: 0 for cid in self.class_ids}