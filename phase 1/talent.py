"""
talent_model.py
----------------
Phase 1: Text-based Sports Talent Identification

Responsibilities:
- Feature engineering
- Normalization
- Talent scoring
- Sport recommendation
- Football position recommendation

Author: AI Sports Analytics System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TalentIdentifier:
    """
    A lightweight and explainable talent identification model
    based on structured player data.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    # -----------------------------------
    # Data Cleaning
    # -----------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans raw player data.

        - Removes missing values
        - Filters unrealistic ages

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df = df.dropna()
        df = df[df["age"].between(10, 40)]
        return df

    # -----------------------------------
    # Feature Engineering
    # -----------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new performance-related features.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()

        df["BMI"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

        df["physical_index"] = df[
            ["speed", "strength", "stamina"]
        ].mean(axis=1)

        df["technical_index"] = df[
            ["passing", "shooting"]
        ].mean(axis=1)

        df["agility_score"] = df["agility"] * df["speed"]

        return df

    # -----------------------------------
    # Normalization
    # -----------------------------------
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes engineered features for fair comparison.

        Returns
        -------
        pd.DataFrame
        """
        features = df[
            ["physical_index", "technical_index", "agility_score"]
        ]

        scaled = self.scaler.fit_transform(features)

        df["phys_norm"] = scaled[:, 0]
        df["tech_norm"] = scaled[:, 1]
        df["agil_norm"] = scaled[:, 2]

        return df

    # -----------------------------------
    # Talent Score
    # -----------------------------------
    def compute_talent_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes final talent score using an explainable formula.

        Returns
        -------
        pd.DataFrame
        """
        df["talent_score"] = (
            0.35 * df["phys_norm"]
            + 0.30 * df["tech_norm"]
            + 0.20 * df["agil_norm"]
            + 0.15 * (1 - abs(df["age"] - 22) / 22)
        )

        return df

    # -----------------------------------
    # Sport Recommendation
    # -----------------------------------
    @staticmethod
    def recommend_sport(row) -> str:
        """
        Rule-based sport recommendation.
        """
        if row["strength"] > 80 and row["BMI"] > 25:
            return "Wrestling"
        elif row["speed"] > 80 and row["agility"] > 75:
            return "Football"
        elif row["stamina"] > 85:
            return "Athletics"
        else:
            return "Football"

    # -----------------------------------
    # Football Position Recommendation
    # -----------------------------------
    @staticmethod
    def football_position(row) -> str:
        """
        Rule-based football position recommendation.
        """
        if row["shooting"] > 80:
            return "Striker"
        elif row["passing"] > 80 and row["stamina"] > 75:
            return "Midfielder"
        elif row["strength"] > 80 and row["height_cm"] > 180:
            return "Defender"
        elif row["speed"] > 85:
            return "Winger"
        else:
            return "Substitute"

    # -----------------------------------
    # Full Pipeline
    # -----------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full Phase 1 pipeline.

        Returns
        -------
        pd.DataFrame
        """
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.normalize(df)
        df = self.compute_talent_score(df)

        df["recommended_sport"] = df.apply(
            self.recommend_sport, axis=1
        )

        df["football_position"] = df.apply(
            self.football_position, axis=1
        )

        df = df.sort_values("talent_score", ascending=False)

        return df
