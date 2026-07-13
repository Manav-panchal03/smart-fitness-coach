import streamlit as st
import joblib
import numpy as np

# loading models - saved!
model_simple = joblib.load('calorie_model.pkl')
model_multi = joblib.load('calories_model_multi.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
cluster_names = joblib.load('cluster_names.pkl')

st.title("🏋️ Smart Fitness Coach")
st.write("Predict your calories burned and discover your training persona.")

st.header("Predict Your Calories Burned")

model_choice = st.radio(
    "Choose prediction mode:",
    ["Quick (Session Duration only)", "Detailed (Duration + Experience + Fat%)"]
)

if model_choice == "Quick (Session Duration only)":
    st.caption("ℹ️ We'll assume Experience Level = 2 and Fat Percentage = 26.2% (dataset averages) for your persona detection. For a more accurate, personalized result based on your real stats, switch to **Detailed** mode.")

duration = st.slider("Session Duration (hours)", min_value=0.25, max_value=2.5, value=1.0, step=0.05)
workout_frequency = st.slider("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)

if model_choice == "Detailed (Duration + Experience + Fat%)":
    experience = st.selectbox("Experience Level", [1, 2, 3])
    fat_percentage = st.slider("Fat Percentage (%)", min_value=5.0, max_value=45.0, value=20.0, step=0.5)

if st.button("Predict"):
    # --- Calorie prediction ---
    if model_choice == "Quick (Session Duration only)":
        input_data = np.array([[duration]])
        prediction = model_simple.predict(input_data)[0]
        experience_for_cluster = 2
        fat_percentage_for_cluster = 26.2
    else:
        input_data = np.array([[duration, experience, fat_percentage]])
        prediction = model_multi.predict(input_data)[0]
        experience_for_cluster = experience
        fat_percentage_for_cluster = fat_percentage

    st.success(f"Estimated Calories Burned: {prediction:.0f} kcal")

    # --- Persona detection ---
    cluster_input = np.array([[duration, prediction, workout_frequency,
                                experience_for_cluster, fat_percentage_for_cluster]])
    cluster_input_scaled = scaler.transform(cluster_input)
    cluster_num = kmeans.predict(cluster_input_scaled)[0]
    persona = cluster_names[cluster_num]

    st.info(f"Your Training Persona: **{persona}**")