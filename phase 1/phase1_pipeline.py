import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("/mnt/data/players.csv")

# ----------------------------
# Basic Cleaning
# ----------------------------
df = df.dropna()
df = df[df["age"].between(10, 40)]

# ----------------------------
# Feature Engineering
# ----------------------------
df["BMI"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

df["physical_index"] = df[["speed", "strength", "stamina"]].mean(axis=1)
df["technical_index"] = df[["passing", "shooting"]].mean(axis=1)
df["agility_score"] = df["agility"] * df["speed"]

# ----------------------------
# Normalization
# ----------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    df[["physical_index", "technical_index", "agility_score"]]
)

df[["phys_norm", "tech_norm", "agil_norm"]] = scaled_features

# ----------------------------
# Talent Score
# ----------------------------
df["talent_score"] = (
    0.35 * df["phys_norm"] +
    0.30 * df["tech_norm"] +
    0.20 * df["agil_norm"] +
    0.15 * (1 - abs(df["age"] - 22) / 22)
)

# ----------------------------
# Sport Recommendation
# ----------------------------
def recommend_sport(row):
    if row["strength"] > 80 and row["BMI"] > 25:
        return "Wrestling"
    elif row["speed"] > 80 and row["agility"] > 75:
        return "Football"
    elif row["stamina"] > 85:
        return "Athletics"
    else:
        return "Football"

df["recommended_sport"] = df.apply(recommend_sport, axis=1)

# ----------------------------
# Football Position
# ----------------------------
def football_position(row):
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

df["football_position"] = df.apply(football_position, axis=1)

# ----------------------------
# Rank Players
# ----------------------------
df = df.sort_values("talent_score", ascending=False)

print(df[[
    "age", "height_cm", "weight_kg",
    "talent_score",
    "recommended_sport",
    "football_position"
]].head(10))
