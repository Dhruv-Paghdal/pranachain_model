import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="CKD Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .low-risk {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .high-risk {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .risk-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .risk-probability {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    .risk-factor {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .risk-factor-red {
        background-color: #fff5f5;
        border-color: #dc3545;
    }
    .risk-factor-yellow {
        background-color: #fffdf0;
        border-color: #ffc107;
    }
    .risk-factor-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    .risk-factor-detail {
        color: #666;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


def calculate_ckd_stage(df):
    g_stage = np.select(
        [
            df['GFR'] >= 90,
            (df['GFR'] >= 60) & (df['GFR'] < 90),
            (df['GFR'] >= 45) & (df['GFR'] < 60),
            (df['GFR'] >= 30) & (df['GFR'] < 45),
            (df['GFR'] >= 15) & (df['GFR'] < 30),
            df['GFR'] < 15
        ],
        [1, 2, 3, 4, 5, 5],
        default=0
    )
    
    a_stage = np.select(
        [
            df['ACR'] < 30,
            (df['ACR'] >= 30) & (df['ACR'] <= 300),
            df['ACR'] > 300
        ],
        [1, 2, 3],
        default=0
    )
    
    df['ckd_stage'] = g_stage * 10 + a_stage
    return df


def calculate_htn_dm(df):
    df['htn'] = ((df['SystolicBP'] >= 140) | (df['DiastolicBP'] >= 90)).astype(int)
    df['dm'] = ((df['FastingBloodSugar'] >= 126) | (df['HbA1c'] >= 6.5)).astype(int)
    return df


def parse_report_file(file_content):
    lines = file_content.strip().split('\n')
    data = {}
    
    field_mapping = {
        'age': 'age',
        'systolicBP': 'SystolicBP',
        'diastolicBP': 'DiastolicBP',
        'fastingBloodSugar': 'FastingBloodSugar',
        'hbA1c': 'HbA1c',
        'serumCreatinine': 'sc',
        'gfr': 'GFR',
        'hemoglobin': 'hemo',
        'cholesterolTotal': 'CholesterolTotal',
        'cholesterolLDL': 'CholesterolLDL',
        'cholesterolHDL': 'CholesterolHDL',
        'cholesterolTriglycerides': 'CholesterolTriglycerides',
        'proteinInUrine': 'al',
        'acr': 'ACR'
    }
    
    for line in lines:
        if '=' in line:
            key, value = line.strip().split('=')
            if key in field_mapping:
                data[field_mapping[key]] = float(value)
    
    return data


def predict_ckd_risk(patient_data, model_bundle):
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    optimal_threshold = model_bundle['optimal_threshold']
    cols_to_scale = model_bundle['cols_to_scale']
    ckd_features = model_bundle['ckd_features']
    
    try:
        data_df = pd.DataFrame([patient_data], columns=ckd_features)
        
        data_df = calculate_ckd_stage(data_df)
        data_df = calculate_htn_dm(data_df)
        
        data_scaled = data_df.copy().astype(float)
        if cols_to_scale:
            data_scaled.loc[:, cols_to_scale] = scaler.transform(data_df[cols_to_scale])
        
        y_pred_proba = model.predict_proba(data_scaled)[:, 1]
        risk_score = y_pred_proba[0]
        prediction_class = int(risk_score >= optimal_threshold)
        
        return risk_score, prediction_class, data_df
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None


def identify_risk_factors(patient_data, data_df):
    risk_factors = []
    
    # Check Hypertension
    if patient_data['SystolicBP'] >= 140 or patient_data['DiastolicBP'] >= 90:
        risk_factors.append({
            'title': 'Hypertension',
            'detail': f"Blood pressure is elevated ({patient_data['SystolicBP']:.0f}/{patient_data['DiastolicBP']:.0f} mmHg). Normal range: <120/80 mmHg",
            'severity': 'high'
        })
    
    # Check Diabetes
    if patient_data['FastingBloodSugar'] >= 126 or patient_data['HbA1c'] >= 6.5:
        risk_factors.append({
            'title': 'Diabetes',
            'detail': f"Blood sugar levels indicate diabetes (FBS: {patient_data['FastingBloodSugar']:.0f} mg/dL, HbA1c: {patient_data['HbA1c']:.1f}%). Target: FBS <100, HbA1c <5.7%",
            'severity': 'high' if patient_data['HbA1c'] >= 7.0 else 'moderate'
        })
    
    # Check Kidney Function
    if patient_data['GFR'] < 60 or patient_data['sc'] > 1.2:
        risk_factors.append({
            'title': 'Kidney Function',
            'detail': f"Kidney function is impaired (Creatinine: {patient_data['sc']:.1f} mg/dL, GFR: {patient_data['GFR']:.0f}). Normal GFR: >90",
            'severity': 'high' if patient_data['GFR'] < 45 else 'moderate'
        })
    
    # Check Anemia
    if patient_data['hemo'] < 12:
        risk_factors.append({
            'title': 'Anemia',
            'detail': f"Hemoglobin is low ({patient_data['hemo']:.1f} g/dL). Normal range: 12-16 g/dL",
            'severity': 'moderate'
        })
    
    # Check Cholesterol
    if patient_data['CholesterolTotal'] >= 200 or patient_data['CholesterolLDL'] >= 100:
        risk_factors.append({
            'title': 'High Cholesterol',
            'detail': f"Cholesterol levels are elevated (Total: {patient_data['CholesterolTotal']:.0f}, LDL: {patient_data['CholesterolLDL']:.0f}). Target: Total <200, LDL <100",
            'severity': 'moderate'
        })
    
    # Check Proteinuria
    if patient_data['ACR'] >= 30 or patient_data['al'] >= 1:
        risk_factors.append({
            'title': 'Proteinuria',
            'detail': f"Protein in urine is elevated (ACR: {patient_data['ACR']:.0f} mg/g). Normal: <30 mg/g",
            'severity': 'high' if patient_data['ACR'] > 300 else 'moderate'
        })
    
    return risk_factors


def main():
    st.markdown('<div class="main-header"> CKD Risk Assessment System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header(" Configuration")
        model_file = st.text_input("Model File Path", value="ckd_stacking_model.joblib")
        
        if st.button("Load Model"):
            if os.path.exists(model_file):
                try:
                    st.session_state['model_bundle'] = joblib.load(model_file)
                    st.success(" Model loaded successfully!")
                except Exception as e:
                    st.error(f" Error loading model: {str(e)}")
            else:
                st.error(f" Model file '{model_file}' not found!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Upload Patient Report")
        uploaded_file = st.file_uploader("Choose a report file", type=['txt'])
        
        if uploaded_file is not None:
            # Read and display file content
            file_content = uploaded_file.read().decode('utf-8')
            st.text_area("Report Content", file_content, height=300)
            
            # Parse the report
            try:
                patient_data = parse_report_file(file_content)
                st.success(" Report parsed successfully!")
                
                # Display parsed data
                with st.expander("View Parsed Data"):
                    st.json(patient_data)
                
            except Exception as e:
                st.error(f" Error parsing report: {str(e)}")
                patient_data = None
        else:
            st.info(" Please upload a patient report file to begin analysis")
            patient_data = None
    
    with col2:
        st.subheader(" Analysis Results")
        
        if st.button(" Analyze Report", type="primary", disabled=(uploaded_file is None)):
            if 'model_bundle' not in st.session_state:
                st.warning(" Please load the model first from the sidebar!")
            elif patient_data is not None:
                with st.spinner("Analyzing patient data..."):
                    risk_score, prediction, data_df = predict_ckd_risk(
                        patient_data, 
                        st.session_state['model_bundle']
                    )
                    
                    if risk_score is not None:
                        st.session_state['results'] = {
                            'risk_score': risk_score,
                            'prediction': prediction,
                            'patient_data': patient_data,
                            'data_df': data_df
                        }
    
    if 'results' in st.session_state:
        st.markdown("---")
        results = st.session_state['results']
        risk_score = results['risk_score']
        prediction = results['prediction']
        patient_data = results['patient_data']
        
        if prediction == 0:
            st.markdown(f"""
            <div class="risk-box low-risk">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 2rem; margin-right: 15px;"></span>
                    <span class="risk-title">Low Risk</span>
                </div>
                <div class="risk-probability">CKD Probability: {risk_score*100:.1f}%</div>
                <p style="margin-top: 15px; color: #155724;">
                    The analysis shows a low risk of chronic kidney disease. Continue maintaining healthy lifestyle habits.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-box high-risk">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 2rem; margin-right: 15px;"></span>
                    <span class="risk-title">High Risk</span>
                </div>
                <div class="risk-probability">CKD Probability: {risk_score*100:.1f}%</div>
                <p style="margin-top: 15px; color: #721c24;">
                    The analysis indicates an elevated risk of chronic kidney disease. Please consult with a healthcare provider for further evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Factors Section
        st.markdown("###  Identified Risk Factors")
        risk_factors = identify_risk_factors(patient_data, results['data_df'])
        
        if risk_factors:
            for factor in risk_factors:
                severity_class = 'risk-factor-red' if factor['severity'] == 'high' else 'risk-factor-yellow'
                icon = '❤️' if factor['severity'] == 'high' else '⚠️'
                
                st.markdown(f"""
                <div class="risk-factor {severity_class}">
                    <div class="risk-factor-title">{icon} {factor['title']}</div>
                    <div class="risk-factor-detail">{factor['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success(" No significant risk factors identified.")
        
        # Detailed Metrics
        st.markdown("---")
        st.markdown("###  Detailed Patient Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Age", f"{patient_data['age']:.0f} years")
            st.metric("Systolic BP", f"{patient_data['SystolicBP']:.0f} mmHg")
            st.metric("Hemoglobin", f"{patient_data['hemo']:.1f} g/dL")
        
        with metric_col2:
            st.metric("GFR", f"{patient_data['GFR']:.0f}")
            st.metric("Diastolic BP", f"{patient_data['DiastolicBP']:.0f} mmHg")
            st.metric("Creatinine", f"{patient_data['sc']:.1f} mg/dL")
        
        with metric_col3:
            st.metric("FBS", f"{patient_data['FastingBloodSugar']:.0f} mg/dL")
            st.metric("HbA1c", f"{patient_data['HbA1c']:.1f}%")
            st.metric("ACR", f"{patient_data['ACR']:.0f} mg/g")
        
        with metric_col4:
            st.metric("Total Chol", f"{patient_data['CholesterolTotal']:.0f}")
            st.metric("LDL", f"{patient_data['CholesterolLDL']:.0f}")
            st.metric("HDL", f"{patient_data['CholesterolHDL']:.0f}")


if __name__ == "__main__":
    main()