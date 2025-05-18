import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import joblib
import os

# Set page config must be the first Streamlit command
st.set_page_config(page_title="AI Health Advisor", page_icon="🧠", layout="wide")

# Load the actual trained model and scaler
@st.cache_resource
def load_models():
    try:
        # Try to load pre-trained model files
        if os.path.exists('diabetes_model.h5') and os.path.exists('scaler.pkl'):
            from tensorflow.keras.models import load_model
            model = load_model('diabetes_model.h5')
            scaler = joblib.load('scaler.pkl')
            st.success("Loaded trained model successfully!")
            return scaler, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    
    # Fallback to creating a simple model if pre-trained files not found
    st.warning("Using dummy model - predictions will not be accurate. Please provide trained model files.")
    
    # Create a dummy scaler
    scaler = StandardScaler()
    dummy_data = np.random.rand(100, 10)
    scaler.fit(dummy_data)
    
    # Create a simple model architecture
    def build_mlp_model(input_dim):
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    model = build_mlp_model(10)
    return scaler, model

scaler, model = load_models()

# History storage
if 'history_log' not in st.session_state:
    st.session_state.history_log = []

def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    try:
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        smoking_map = {
            'never': 0,
            'No Info': 1,
            'current': 2,
            'former': 3,
            'ever': 4,
            'not current': 5
        }
        
        gender_num = gender_map[gender]
        smoking_num = smoking_map[smoking_history]
        hypertension_num = 1 if hypertension else 0
        heart_disease_num = 1 if heart_disease else 0

        bmi_age = bmi * age
        glucose_bp = glucose * hypertension_num

        input_data = np.array([[
            gender_num, age, hypertension_num, heart_disease_num,
            smoking_num, bmi, hba1c, glucose, bmi_age, glucose_bp
        ]])

        input_scaled = scaler.transform(input_data)
        prob = float(model.predict(input_scaled, verbose=0)[0][0])
        prediction = "Positive" if prob > 0.5 else "Negative"

        risks = []
        if hypertension_num: 
            risks.append("• Hypertension increases risk")
        if heart_disease_num: 
            risks.append("• Heart disease increases risk")
        if bmi > 30: 
            risks.append(f"• High BMI ({bmi}) indicates obesity risk")
        if glucose > 140: 
            risks.append(f"• High glucose level ({glucose} mg/dL)")
        if hba1c > 6.5: 
            risks.append(f"• HbA1c ({hba1c}) suggests possible diabetes")

        risk_analysis = "\n".join(risks) if risks else "No significant risk factors"

        st.session_state.history_log.append({
            "Gender": gender,
            "Age": age,
            "Hypertension": "Yes" if hypertension else "No",
            "Heart Disease": "Yes" if heart_disease else "No",
            "Smoking": smoking_history,
            "BMI": bmi,
            "HbA1c": hba1c,
            "Glucose": glucose,
            "Prediction": prediction,
            "Probability": f"{prob:.1%}"
        })

        return prediction, f"{prob:.1%}", risk_analysis

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", "Error", f"Error: {str(e)}"

# Header
st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(to right, #a1c4fd, #c2e9fb); 
    border-radius: 12px; margin-bottom: 20px;">
        <h1 style="font-size: 2.5em; color: #2c3e50;">🧠 AI Health Advisor</h1>
        <p style="font-size: 1.2em; color: black; margin-top: -10px; font-weight: 600;">Diabetes Risk Prediction</p>
        <p style="font-size: 1.2em; color: #34495e;">Clinical precision AI for diabetes assessment</p>
    </div>
""", unsafe_allow_html=True)

# Main columns
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
    bmi = st.slider("BMI", 10.0, 70.0, 25.0, 0.1)
    hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5, 0.1)
    glucose = st.slider("Blood Glucose Level", 50, 300, 100, 1)

st.markdown("---")

# Prediction button
if st.button("🚀 Predict Diabetes Risk", type="primary"):
    with st.spinner('Analyzing health data...'):
        prediction, probability, risk = predict_diabetes(
            gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose
        )
    
    # Show results in columns
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("### 📊 Prediction Results")
        st.text_input("🩺 Prediction", prediction, disabled=True)
        st.text_input("📈 Probability", probability, disabled=True)
        st.text_area("⚠️ Risk Analysis", risk, height=150, disabled=True)
    
    with res_col2:
        st.markdown("### 🗂️ Latest Prediction")
        if st.session_state.history_log:
            latest = st.session_state.history_log[-1]
            st.json({
                "Gender": latest["Gender"],
                "Age": latest["Age"],
                "BMI": latest["BMI"],
                "Glucose": latest["Glucose"],
                "Prediction": latest["Prediction"],
                "Probability": latest["Probability"]
            })

# History section
st.markdown("---")
st.markdown("### 📜 Prediction History")
if st.button("View Full History"):
    if not st.session_state.history_log:
        st.warning("No predictions made yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(history_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 0.9em; color: #95a5a6;">
        © 2025 AI Health Advisor · For educational purposes only
    </div>
""", unsafe_allow_html=True)