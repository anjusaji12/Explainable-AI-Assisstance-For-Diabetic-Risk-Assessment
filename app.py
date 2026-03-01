#stramlit



import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clinical Assistant", layout="wide")

@st.cache_resource
def load_data_and_model():
    # Load and preprocess (Simplified for the demo)
    data = pd.read_csv(r'D:\project\diabetes_012_health_indicators_BRFSS2015.csv')
    data = data.drop(['Education','Fruits','Veggies','Sex','Income'], axis=1)
    
    X = data.drop('Diabetes_012', axis=1)
    Y = data['Diabetes_012']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)
    
    # Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        random_state=42
    )
    model.fit(X_train_balanced, Y_train_balanced)
    
    # SHAP Explainer (Using background data)
    def predict_diabetes_prob(x):
        return model.predict_proba(x)[:, 2]
    
    explainer = shap.Explainer(predict_diabetes_prob, X_train_balanced.iloc[:100, :])
    
    return model, explainer, X.columns

model, explainer, feature_names = load_data_and_model()

# --- SIDEBAR: PATIENT INPUT ---
st.sidebar.header("Patient Clinical Data")

def user_input_features():
    inputs = {}
    inputs['HighBP'] = st.sidebar.selectbox("High Blood Pressure", [0.0, 1.0])
    inputs['HighChol'] = st.sidebar.selectbox("High Cholesterol", [0.0, 1.0])
    inputs['CholCheck'] = st.sidebar.selectbox("Cholesterol Check (5yrs)", [0.0, 1.0])
    inputs['BMI'] = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    inputs['Smoker'] = st.sidebar.selectbox("Smoker", [0.0, 1.0])
    inputs['Stroke'] = st.sidebar.selectbox("History of Stroke", [0.0, 1.0])
    inputs['HeartDiseaseorAttack'] = st.sidebar.selectbox("Heart Disease/Attack", [0.0, 1.0])
    inputs['PhysActivity'] = st.sidebar.selectbox("Physical Activity", [0.0, 1.0])
    inputs['HvyAlcoholConsump'] = st.sidebar.selectbox("Heavy Alcohol", [0.0, 1.0])
    inputs['AnyHealthcare'] = st.sidebar.selectbox("Has Healthcare", [0.0, 1.0])
    inputs['NoDocbcCost'] = st.sidebar.selectbox("No Doctor due to Cost", [0.0, 1.0])
    inputs['GenHlth'] = st.sidebar.slider("General Health (1=Ex, 5=Poor)", 1.0, 5.0, 3.0)
    inputs['MentHlth'] = st.sidebar.slider("Mental Health (Days unwell)", 0.0, 30.0, 0.0)
    inputs['PhysHlth'] = st.sidebar.slider("Physical Health (Days unwell)", 0.0, 30.0, 0.0)
    inputs['DiffWalk'] = st.sidebar.selectbox("Difficulty Walking", [0.0, 1.0])
    inputs['Age'] = st.sidebar.slider("Age Category (1-13)", 1.0, 13.0, 7.0)
    
    return pd.DataFrame([inputs])

input_df = user_input_features()

# --- MAIN PANEL ---
st.title(" Clinical Assistant")
st.markdown("""
This tool uses a **machine learning model (XGBoost)** to predict diabetic risk and **SHAP (Explainable AI)** to show why a specific prediction was made.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Diagnostic Prediction")
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    status_map = {0: "Healthy", 1: "Pre-Diabetic", 2: "Diabetic"}
    color_map = {0: "green", 1: "orange", 2: "red"}
    
    st.markdown(f"### Status: :{color_map[prediction]}[{status_map[prediction]}]")
    st.metric("Confidence (Diabetic)", f"{probs[2]:.2%}")
    st.write("---")
    st.write("**Full Probabilities:**")
    st.write(pd.DataFrame(probs, index=status_map.values(), columns=["Probability"]))

with col2:
    st.subheader("Explainability: Why this result?")
    st.info("The Waterfall plot below shows how each clinical factor contributed to the 'Diabetic' risk score.")
    
    # Generate SHAP values
    shap_values = explainer(input_df)
    
    # Plotting SHAP
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

st.divider()
#st.caption("Disclaimer: This is a clinical assistance tool for educational purposes and should not replace professional medical judgment.")