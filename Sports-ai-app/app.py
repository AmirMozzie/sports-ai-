import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Sports Talent Identification",
    layout="wide"
)

st.title("🏆 AI-Based Sports Talent Identification System")

# =====================================================
# PHASE 1 — CSV TALENT ANALYSIS
# =====================================================
st.header("🟢 Phase 1: Talent Identification (CSV Data)")

csv_file = st.file_uploader("Upload players CSV file", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # ---------- Cleaning ----------
    df = df.dropna()
    df = df[df["age"].between(10, 40)]

    # ---------- Feature Engineering ----------
    df["BMI"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    df["physical_index"] = df[["speed", "strength", "stamina"]].mean(axis=1)
    df["technical_index"] = df[["passing", "shooting"]].mean(axis=1)
    df["agility_score"] = df["agility"] * df["speed"]

    # ---------- Normalization ----------
    scaler = StandardScaler()
    df[["phys_n", "tech_n", "agil_n"]] = scaler.fit_transform(
        df[["physical_index", "technical_index", "agility_score"]]
    )

    # ---------- Talent Score ----------
    df["talent_score"] = (
        0.35 * df["phys_n"]
        + 0.30 * df["tech_n"]
        + 0.20 * df["agil_n"]
        + 0.15 * (1 - abs(df["age"] - 22) / 22)
    )

    # ---------- Sport Recommendation ----------
    def recommend_sport(r):
        if r["strength"] > 80 and r["BMI"] > 25:
            return "Wrestling"
        elif r["speed"] > 80 and r["agility"] > 75:
            return "Football"
        elif r["stamina"] > 85:
            return "Athletics"
        else:
            return "Football"

    df["recommended_sport"] = df.apply(recommend_sport, axis=1)

    # ---------- Football Position ----------
    def football_position(r):
        if r["shooting"] > 80:
            return "Striker"
        elif r["passing"] > 80 and r["stamina"] > 75:
            return "Midfielder"
        elif r["strength"] > 80 and r["height_cm"] > 180:
            return "Defender"
        elif r["speed"] > 85:
            return "Winger"
        else:
            return "Substitute"

    df["position"] = df.apply(football_position, axis=1)

    df = df.sort_values("talent_score", ascending=False)

    st.subheader("🏅 Top Talents")
    st.dataframe(
        df[[
            "age", "height_cm", "weight_kg",
            "talent_score",
            "recommended_sport",
            "position"
        ]].head(10)
    )

# =====================================================
# PHASE 2 — POSE DETECTION
# =====================================================
st.header("🔵 Phase 2: Pose & Biomechanics Analysis")

image_file = st.file_uploader("Upload player image", type=["jpg", "png", "jpeg"])

if image_file:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pose.process(img_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            img_rgb,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        st.image(img_rgb, caption="Detected Pose", use_column_width=True)

        landmarks = result.pose_landmarks.landmark

        # Simple biomechanical scores
        left_hip = landmarks[23].x
        right_hip = landmarks[24].x
        balance_score = 1 - abs(left_hip - right_hip)

        left_ankle = landmarks[27].y
        right_ankle = landmarks[28].y
        coordination_score = 1 - abs(left_ankle - right_ankle)

        visual_score = (
            0.5 * balance_score +
            0.5 * coordination_score
        )

        st.metric("🧍 Balance Score", round(balance_score, 2))
        st.metric("🤸 Coordination Score", round(coordination_score, 2))
        st.metric("👁️ Visual Talent Score", round(visual_score, 2))

# =====================================================
# PHASE 3 — MATCH PREDICTION
# =====================================================
st.header("🔴 Phase 3: Match Outcome Prediction")

st.subheader("⚽ Match Inputs")

rating_diff = st.slider("Team Rating Difference", -5.0, 5.0, 0.0)
home_advantage = st.selectbox("Home Advantage", [0, 1])
recent_form = st.slider("Recent Form (last 5 matches)", 0, 5, 3)
visual_diff = st.slider("Visual Score Difference", -1.0, 1.0, 0.0)

# Dummy trained logistic model
X_train = np.array([
    [-2, 0, 1, -0.3],
    [2, 1, 4, 0.5],
    [1, 1, 3, 0.2],
    [-1, 0, 2, -0.1],
    [3, 1, 5, 0.7],
])
y_train = np.array([0, 1, 1, 0, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

X_input = np.array([[rating_diff, home_advantage, recent_form, visual_diff]])
prob = model.predict_proba(X_input)[0][1]

st.subheader("📈 Match Prediction Result")
st.metric("🏆 Win Probability (Team A)", f"{prob*100:.1f}%")

st.info(
    "Explanation:\n"
    "- Higher team rating difference increases win chance\n"
    "- Home advantage adds positive bias\n"
    "- Recent form and visual performance influence prediction"
)

# =====================================================
st.success("✅ System ready for demo and submission")
