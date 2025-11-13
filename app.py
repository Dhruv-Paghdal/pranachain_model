import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Caching the Model and Scaler ---
@st.cache_resource
def load_assets():
    """Load the model and scaler from disk."""
    try:
        model = joblib.load("diabetes_model.joblib")
        scaler = joblib.load("diabetes_scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or Scaler file not found.")
        st.error("Please make sure 'diabetes_model.joblib' and 'diabetes_scaler.joblib' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the assets: {e}")
        return None, None

model, scaler = load_assets()

# --- Main Application ---
if model is not None and scaler is not None:
    st.title("🩺 Diabetes Risk Predictor")
    st.write("""
    Enter the patient's information in the sidebar to predict their risk of diabetes.
    This prediction is based on a machine learning model and is not a substitute for professional medical advice.
    """)

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Patient Information")

    # CORRECTED: This matches the exact order from your training notebook
    feature_names = [
        'Age',
        'Gender',
        'BMI',
        'SystolicBP',
        'DiastolicBP',
        'A1c',
        'Glucose',
        'TotalCholesterol',
        'Triglycerides',
        'FamilyHistory'
    ]

    # --- Input Fields ---
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    
    # Categorical inputs
    gender_str = st.sidebar.selectbox("Gender", ("Male", "Female"))
    
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
    systolic_bp = st.sidebar.number_input("Systolic BP (mm Hg)", min_value=60, max_value=200, value=120)
    diastolic_bp = st.sidebar.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=140, value=80)
    a1c = st.sidebar.number_input("A1c Level (%)", min_value=4.0, max_value=16.0, value=5.7, format="%.1f")
    glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=100)
    total_chol = st.sidebar.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    triglycerides = st.sidebar.number_input("Triglycerides (mg/dL)", min_value=30, max_value=500, value=150)
    
    family_hist_str = st.sidebar.selectbox("Family History of Diabetes", ("No", "Yes"))

    # --- Prediction Button ---
    if st.sidebar.button("Predict Diabetes Risk", type="primary"):
        
        # Map string inputs to numerical values
        gender_val = 1 if gender_str == "Male" else 0
        family_hist_val = 1 if family_hist_str == "Yes" else 0

        # CORRECTED: Collect inputs in the EXACT same order as training
        user_input_list = [
            age,
            gender_val,
            bmi,
            systolic_bp,
            diastolic_bp,
            a1c,
            glucose,
            total_chol,
            triglycerides,
            family_hist_val
        ]

        # Create a DataFrame
        input_data = pd.DataFrame([user_input_list], columns=feature_names)

        try:
            # Scale the user's input data
            scaled_input_data = scaler.transform(input_data)

            # Make the prediction
            prediction = model.predict(scaled_input_data)
            prediction_proba = model.predict_proba(scaled_input_data)[0]

            # Display the result
            st.header("Prediction Result")
            
            probability_of_diabetes = prediction_proba[1] 

            if prediction[0] == 1:
                st.error(f"**High Risk of Diabetes**")
                st.metric(label="Predicted Probability", value=f"{probability_of_diabetes * 100:.1f}%")
            else:
                st.success(f"**Low Risk of Diabetes**")
                st.metric(label="Predicted Probability", value=f"{probability_of_diabetes * 100:.1f}%")
            
            # Show a progress bar
            st.progress(probability_of_diabetes)
            
            st.subheader("What does this mean?")
            st.write(f"""
            The model predicts a **{probability_of_diabetes * 100:.1f}%** probability of having diabetes.
            
            - A **Low Risk** result indicates that, based on the provided data, you are unlikely to have diabetes.
            - A **High Risk** result suggests you should consult a healthcare professional for further testing and advice.
            """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please ensure all input values are correctly formatted.")

else:
    st.error("Application cannot start. Model or scaler assets failed to load.")