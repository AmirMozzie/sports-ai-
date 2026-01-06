"""
biomechanics.py
----------------
Phase 2: Biomechanical Feature Engineering

Responsibilities:
- Compute balance
- Compute coordination
- Compute posture
- Produce visual talent score
"""

import numpy as np


class BiomechanicsAnalyzer:
    """
    Extracts biomechanical scores from pose landmarks.
    """

    @staticmethod
    def balance_score(landmarks) -> float:
        """
        Balance based on hip symmetry.
        """
        left_hip = landmarks[23].x
        right_hip = landmarks[24].x
        return max(0.0, 1 - abs(left_hip - right_hip))

    @staticmethod
    def coordination_score(landmarks) -> float:
        """
        Coordination based on ankle symmetry.
        """
        left_ankle = landmarks[27].y
        right_ankle = landmarks[28].y
        return max(0.0, 1 - abs(left_ankle - right_ankle))

    @staticmethod
    def posture_score(landmarks) -> float:
        """
        Posture based on shoulder–hip alignment.
        """
        shoulder_center = (landmarks[11].y + landmarks[12].y) / 2
        hip_center = (landmarks[23].y + landmarks[24].y) / 2
        return max(0.0, 1 - abs(shoulder_center - hip_center))

    def visual_talent_score(self, landmarks) -> dict:
        """
        Final visual talent score.

        Returns
        -------
        dict
        """
        balance = self.balance_score(landmarks)
        coordination = self.coordination_score(landmarks)
        posture = self.posture_score(landmarks)

        visual_score = (
            0.4 * balance +
            0.35 * coordination +
            0.25 * posture
        )

        return {
            "balance_score": round(balance, 3),
            "coordination_score": round(coordination, 3),
            "posture_score": round(posture, 3),
            "visual_talent_score": round(visual_score, 3)
        }
