# 🏋️ Smart Fitness Coach

A machine learning web app that predicts calories burned during a workout and identifies a user's training persona, built on real gym member data and deployed live with Streamlit.

**[🔗 Live Demo](#)** *(add your Streamlit Cloud link here once confirmed live)*

---

## Problem Statement

Gym members often don't know how their workout habits compare to others, or what factors actually drive calorie burn during a session. This project explores: **what factors most influence calories burned during a workout, and can gym-goers be grouped into meaningful, distinct training personas based on their behavior?**

---

## Dataset

- **Source:** [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) (Kaggle)
- **Size:** 973 records, 15 features
- **Features used:** Age, Weight, Height, BMI, Session Duration, Workout Frequency, Experience Level, Fat Percentage, Heart Rate metrics, Calories Burned, Workout Type

---

## Approach

1. **Exploratory Data Analysis** — examined distributions, checked for missing values and outliers (found none — clean dataset), and used `.describe()` to understand each feature's spread.
2. **Correlation Analysis** — built a correlation matrix to identify which features actually relate to Calories Burned, and checked for multicollinearity between predictor variables.
3. **Regression Modeling** — built two Linear Regression models:
   - Simple model (Session Duration only)
   - Multi-feature model (Session Duration + Experience Level + Fat Percentage)
4. **Clustering** — used K-Means (with the Elbow Method to select K=4) to segment gym members into four distinct training personas based on behavioral features.
5. **Deployment** — built an interactive Streamlit app that loads the trained models and lets users get real-time predictions and persona classification, with a Quick/Detailed mode toggle.

---

## Key Findings

- **Session Duration is the dominant driver of calories burned**, with a correlation of 0.91 and R² of 0.82 on its own — the strongest single relationship in the dataset.
- **Adding more features barely improved the model** (R² moved from 0.8168 → 0.8207), showing that more inputs don't always mean a meaningfully better model.
- **Experience Level's effect became unstable (and even flipped sign) in the multi-feature model** due to multicollinearity with Session Duration — a good example of why correlated predictors need careful interpretation, not blind inclusion.
- **Clustering revealed two personas with nearly identical session duration and calories burned, but very different workout frequency and experience** ("Occasional Intense Trainers" vs. "Consistent Intermediate Trainers") — a distinction that correlation analysis alone would have missed, since it only shows up when multiple behavioral features are considered together.

### The Four Training Personas

| Persona | Duration (hrs) | Calories | Frequency (days/wk) | Experience | Fat % |
|---|---|---|---|---|---|
| Elite Trainer | 1.76 | 1265 | 4.5 | 3 | 14.8% |
| Occasional Intense Trainer | 1.26 | 919 | 2.5 | 1 | 27.8% |
| Consistent Intermediate Trainer | 1.25 | 902 | 3.5 | 2 | 27.3% |
| Beginner / Light Trainer | 0.77 | 542 | 2.5 | 1 | 27.5% |

---

## App Design Notes

- Users choose between **Quick** mode (Session Duration only) and **Detailed** mode (adds Experience Level + Fat Percentage) for the calorie estimate.
- Persona detection always requires all 5 clustering features. In Quick mode, Experience Level and Fat Percentage default to dataset averages (2 and 26.2% respectively), and the app clearly tells the user this upfront so they know the persona result is an estimate unless they switch to Detailed mode.

---

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Linear Regression, K-Means, StandardScaler)
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Model Persistence:** Joblib

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/smart-fitness-coach.git
cd smart-fitness-coach

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Project Structure

```
smart-fitness-coach/
│
├── app.py                              # Streamlit app
├── analysis.py                         # Full EDA, modeling, and clustering workflow
├── calorie_model.pkl                   # Simple regression model
├── calories_model_multi.pkl            # Multi-feature regression model
├── scaler.pkl                          # StandardScaler for clustering input
├── kmeans_model.pkl                    # Trained K-Means model
├── cluster_names.pkl                   # Cluster number → persona name mapping
├── gym_members_exercise_tracking.csv   # Dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Author

Built by Manav J Panchal as part of a self-directed, project-based data science learning path.
