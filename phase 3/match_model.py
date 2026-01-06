"""
match_model.py
--------------
Phase 3: Match Outcome Prediction

Responsibilities:
- Train a lightweight match prediction model
- Predict win probability
- Provide explainable outputs

Model:
- Logistic Regression (interpretable & fast)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class MatchOutcomePredictor:
    """
    Predicts match outcomes using simple, explainable features.
    """

    def __init__(self):
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """
        Train the model using historical (or synthetic) match data.

        Expected columns:
        - rating_diff
        - home_advantage
        - recent_form
        - visual_score_diff
        - match_outcome
        """
        X = df[
            ["rating_diff", "home_advantage", "recent_form", "visual_score_diff"]
        ]
        y = df["match_outcome"]

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self,
                rating_diff: float,
                home_advantage: int,
                recent_form: int,
                visual_score_diff: float) -> dict:
        """
        Predict win probability for Team A.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X_input = np.array(
            [[rating_diff, home_advantage, recent_form, visual_score_diff]]
        )

        win_prob = self.model.predict_proba(X_input)[0][1]

        return {
            "win_probability": round(win_prob, 3),
            "loss_probability": round(1 - win_prob, 3)
        }

    def explain(self) -> dict:
        """
        Returns model coefficients for explainability.
        """
        coef = self.model.coef_[0]
        features = [
            "rating_diff",
            "home_advantage",
            "recent_form",
            "visual_score_diff"
        ]

        explanation = dict(zip(features, coef.round(3)))
        return explanation
