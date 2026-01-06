"""
pose_extractor.py
-----------------
Phase 2: Pose Detection using MediaPipe

Responsibilities:
- Load image
- Detect human pose
- Extract body landmarks

Output:
- List of landmarks (x, y, z, visibility)
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseExtractor:
    """
    Lightweight pose extraction using MediaPipe Pose.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.mp_draw = mp.solutions.drawing_utils

    def extract_pose(self, image: np.ndarray):
        """
        Extract pose landmarks from an image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (OpenCV format)

        Returns
        -------
        landmarks : list or None
        annotated_image : np.ndarray
        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image_rgb)

        if not result.pose_landmarks:
            return None, image

        landmarks = result.pose_landmarks.landmark

        # Draw skeleton
        annotated_image = image_rgb.copy()
        self.mp_draw.draw_landmarks(
            annotated_image,
            result.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        return landmarks, annotated_image
