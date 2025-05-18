import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import uuid

# Set page config as the first Streamlit command
st.set_page_config(page_title="AI Health Advisor", page_icon="🧠", layout="wide")

# Rest of the code follows
# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess the dataset (assuming the dataset is available)
@st.cache_data
def load_data():
    file_path = "diabetes_prediction_dataset.csv"  # Update path as needed
    diabetes_data = pd.read_csv(file_path)
    
    # Data Cleaning and Preprocessing
    diabetes_data['gender'] = diabetes_data['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    diabetes_data['smoking_history'] = diabetes_data['smoking_history'].map({
        'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5
    })
    
    # Feature Engineering
    diabetes_data['bmi_age_interaction'] = diabetes_data['bmi'] * diabetes_data['age']
    diabetes_data['glucose_bp_interaction'] = diabetes_data['blood_glucose_level'] * diabetes_data['hypertension']
    
    # Split into features and target
    X = diabetes_data.drop('diabetes', axis=1)
    y = diabetes_data['diabetes']
    return X, y

# Load data
X, y = load_data()

# Initialize scaler
scaler = StandardScaler()
scaler.fit(X)  # Fit scaler on the entire dataset for consistency

# Define the MLP model (relu_adam_dropout configuration)
def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Load or train the model
@st.cache_resource
def load_model():
    input_dim = X.shape[1]
    model = build_mlp_model(input_dim)
    # Note: In a production setting, you would load a pre-trained model
    # For simplicity, we'll assume the model is trained or use a placeholder
    return model

model = load_model()

# Initialize session state for history
if 'history_log' not in st.session_state:
    st.session_state.history_log = []

# Prediction function
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    try:
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
        gender_num = gender_map[gender]
        smoking_num = smoking_map[smoking_history]
        hypertension_num = int(hypertension)
        heart_disease_num = int(heart_disease)

        bmi_age = bmi * age
        glucose_bp = glucose * hypertension_num

        input_data = np.array([[gender_num, age, hypertension_num, heart_disease_num,
                                smoking_num, bmi, hba1c, glucose, bmi_age, glucose_bp]])

        input_scaled = scaler.transform(input_data)
        prob = model.predict(input_scaled, verbose=0)[0][0]
        prediction = "Positive" if prob > 0.5 else "Negative"

        risks = []
        if hypertension_num: risks.append("• Hypertension increases risk")
        if heart_disease_num: risks.append("• Heart disease increases risk")
        if bmi > 30: risks.append(f"• High BMI ({bmi}) indicates obesity risk")
        if glucose > 140: risks.append(f"• High glucose level ({glucose} mg/dL)")
        if hba1c > 6.5: risks.append(f"• HbA1c ({hba1c}) suggests possible diabetes")

        risk_analysis = "\n".join(risks) if risks else "No significant risk factors"

        # Store in history
        st.session_state.history_log.append({
            "Gender": gender,
            "Age": age,
            "Hypertension": hypertension,
            "Heart Disease": heart_disease,
            "Smoking": smoking_history,
            "BMI": bmi,
            "HbA1c": hba1c,
            "Glucose": glucose,
            "Prediction": prediction,
            "Probability": f"{prob:.1%}"
        })

        return f"🩺 {prediction}", f"📊 {prob:.1%}", risk_analysis

    except Exception as e:
        return "Error", "Error", f"Error: {e}"

# View history function
def view_history():
    if not st.session_state.history_log:
        return "⚠️ No predictions made yet."
    history_text = "\n\n".join([
        f"✅ **{entry['Prediction']}** | 🧪 **{entry['Probability']}**\n"
        f"👤 {entry['Gender']}, Age: {entry['Age']}, BMI: {entry['BMI']}, Glucose: {entry['Glucose']}"
        for entry in st.session_state.history_log
    ])
    return history_text

# Streamlit UI
st.set_page_config(page_title="AI Health Advisor", page_icon="🧠", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #a1c4fd, #c2e9fb);
        padding: 2rem;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput, .stSelectbox, .stSlider, .stCheckbox {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #95a5a6;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header'>
        <h1 style='font-size: 2.5em; color: #2c3e50;'>🧠 AI Health Advisor</h1>
        <p style='font-size: 1.2em; color: black; margin-top: -10px; font-weight: 600;'>Ejlal Hameed</p>
        <p style='font-size: 1.2em; color: #34495e;'>Predict diabetes risk with clinical precision and AI insights</p>
    </div>
""", unsafe_allow_html=True)

# Input Form
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Patient Information")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.slider("Age", 0, 120, 30)
    smoking = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])

with col2:
    st.markdown("### 💉 Medical Details")
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    bmi = st.slider("BMI", 10.0, 70.0, 25.0, step=0.1)
    hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
    glucose = st.slider("Blood Glucose Level", 50, 300, 100, step=1)

st.markdown("---")

# Buttons
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    submit_btn = st.button("🚀 Predict Diabetes Risk")
with col_btn2:
    history_btn = st.button("📜 View Prediction History")

# Output Section
if submit_btn:
    prediction, probability, risk = predict_diabetes(
        gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose
    )
    st.markdown("### 📊 Prediction Results")
    st.text_area("🩺 Prediction", value=prediction, height=50, disabled=True)
    st.text_area("📈 Probability", value=probability, height=50, disabled=True)
    st.text_area("⚠️ Risk Analysis", value=risk, height=150, disabled=True)

if history_btn:
    history_output = view_history()
    st.markdown("### 🗂️ Prediction History")
    st.text_area("🧾 Stored History", value=history_output, height=300, disabled=True)

# Footer
st.markdown("""
    <hr>
    <div class='footer'>
        © 2025 AI Health Advisor · Designed for academic and educational purposes
    </div>
""", unsafe_allow_html=True)