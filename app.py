import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
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
    .risk-factor-green {
        background-color: #f0fff4;
        border-color: #28a745;
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

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_models():
    models = {}
    try:
        models['ckd'] = joblib.load('models/kidney.joblib')
        models['diabetes'] = {
            'model': joblib.load('models/diabetes.joblib'),
            'scaler': joblib.load('models/diabetes_sclar.joblib')
        }
        models['heart'] = {
            'model': joblib.load('models/heart.joblib'),
            'scaler': joblib.load('models/heart_scaler.joblib')
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# ==================== CKD FUNCTIONS ====================
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

def parse_ckd_report(file_content):
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
        st.error(f"Error during CKD prediction: {str(e)}")
        return None, None, None

def identify_ckd_risk_factors(patient_data, prediction):
    risk_factors = []
    
    is_abnormal = patient_data['SystolicBP'] >= 140 or patient_data['DiastolicBP'] >= 90
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'Hypertension',
                'detail': f"Blood pressure is elevated ({patient_data['SystolicBP']:.0f}/{patient_data['DiastolicBP']:.0f} mmHg). Normal range: <120/80 mmHg",
                'severity': 'high'
            })
        else:
            risk_factors.append({
                'title': 'Blood Pressure',
                'detail': f"Blood pressure is normal ({patient_data['SystolicBP']:.0f}/{patient_data['DiastolicBP']:.0f} mmHg). Normal range: <120/80 mmHg",
                'severity': 'low'
            })
    
    is_abnormal = patient_data['FastingBloodSugar'] >= 126 or patient_data['HbA1c'] >= 6.5
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'Diabetes',
                'detail': f"Blood sugar levels indicate diabetes (FBS: {patient_data['FastingBloodSugar']:.0f} mg/dL, HbA1c: {patient_data['HbA1c']:.1f}%). Target: FBS <100, HbA1c <5.7%",
                'severity': 'high' if patient_data['HbA1c'] >= 7.0 else 'moderate'
            })
        else:
            risk_factors.append({
                'title': 'Blood Sugar',
                'detail': f"Blood sugar levels are normal (FBS: {patient_data['FastingBloodSugar']:.0f} mg/dL, HbA1c: {patient_data['HbA1c']:.1f}%). Target: FBS <100, HbA1c <5.7%",
                'severity': 'low'
            })
    
    is_abnormal = patient_data['GFR'] < 60 or patient_data['sc'] > 1.2
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'Kidney Function',
                'detail': f"Kidney function is impaired (Creatinine: {patient_data['sc']:.1f} mg/dL, GFR: {patient_data['GFR']:.0f}). Normal GFR: >90",
                'severity': 'high' if patient_data['GFR'] < 45 else 'moderate'
            })
        else:
            risk_factors.append({
                'title': 'Kidney Function',
                'detail': f"Kidney function is healthy (Creatinine: {patient_data['sc']:.1f} mg/dL, GFR: {patient_data['GFR']:.0f}). Normal GFR: >90",
                'severity': 'low'
            })
    
    is_abnormal = patient_data['hemo'] < 12
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'Anemia',
                'detail': f"Hemoglobin is low ({patient_data['hemo']:.1f} g/dL). Normal range: 12-16 g/dL",
                'severity': 'moderate'
            })
        else:
            risk_factors.append({
                'title': 'Hemoglobin',
                'detail': f"Hemoglobin is normal ({patient_data['hemo']:.1f} g/dL). Normal range: 12-16 g/dL",
                'severity': 'low'
            })
    
    is_abnormal = patient_data['CholesterolTotal'] >= 200 or patient_data['CholesterolLDL'] >= 100
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'High Cholesterol',
                'detail': f"Cholesterol levels are elevated (Total: {patient_data['CholesterolTotal']:.0f}, LDL: {patient_data['CholesterolLDL']:.0f}). Target: Total <200, LDL <100",
                'severity': 'moderate'
            })
        else:
            risk_factors.append({
                'title': 'Cholesterol',
                'detail': f"Cholesterol levels are healthy (Total: {patient_data['CholesterolTotal']:.0f}, LDL: {patient_data['CholesterolLDL']:.0f}). Target: Total <200, LDL <100",
                'severity': 'low'
            })
    
    is_abnormal = patient_data['ACR'] >= 30 or patient_data['al'] >= 1
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        if is_abnormal:
            risk_factors.append({
                'title': 'Proteinuria',
                'detail': f"Protein in urine is elevated (ACR: {patient_data['ACR']:.0f} mg/g). Normal: <30 mg/g",
                'severity': 'high' if patient_data['ACR'] > 300 else 'moderate'
            })
        else:
            risk_factors.append({
                'title': 'Urine Protein',
                'detail': f"Protein in urine is normal (ACR: {patient_data['ACR']:.0f} mg/g). Normal: <30 mg/g",
                'severity': 'low'
            })
    
    return risk_factors

# ==================== DIABETES FUNCTIONS ====================
OPTIMAL_THRESHOLD = 0.4
SCALING_COLUMNS = [
    'Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'A1c', 'Glucose', 
    'TotalCholesterol', 'Triglycerides'
]
FINAL_MODEL_FEATURES = [
    'Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'A1c', 'Glucose', 
    'TotalCholesterol', 'Triglycerides', 'Gender_2.0', 'FamilyHistory_2.0'
]

FILE_KEY_MAPPING = {
    'age': ('Age', float),
    'bmi': ('BMI', float), 
    'systolicBP': ('SystolicBP', float),
    'diastolicBP': ('DiastolicBP', float),
    'hbA1c': ('A1c', float),
    'fastingBloodSugar': ('Glucose', float), 
    'cholesterolTotal': ('TotalCholesterol', float), 
    'cholesterolTriglycerides': ('Triglycerides', float), 
    'htn': ('Hypertension', int), 
    'dm': ('PreviousDM', int), 
    'gender': ('Gender', str),
    'familyhistory': ('FamilyHistory', str)
}

def parse_diabetes_report(file_content):
    data = {}
    content = file_content
    
    pattern = re.compile(r'^\s*(\w+\s?\w*)\s*[:=]\s*([\d\.]+|[A-Za-z]+)', re.MULTILINE)
    
    for line in content.splitlines():
        match = re.search(pattern, line)
        if match:
            key_raw = match.group(1).strip().lower()
            value_raw = match.group(2).strip()
            
            key_sanitized = key_raw.replace(' ', '')
            
            if key_sanitized in [k.lower() for k in FILE_KEY_MAPPING.keys()]:
                original_key = next((k for k in FILE_KEY_MAPPING if k.lower() == key_sanitized), None)
                if original_key:
                    data[original_key] = value_raw
    
    if 'bmi' not in data:
        data['bmi'] = 25.0

    final_data = {}
    for key, (feature_name, dtype) in FILE_KEY_MAPPING.items():
        if key in data:
            try:
                final_data[key] = dtype(data[key])
            except ValueError:
                st.error(f"Error: Could not convert value '{data[key]}' for key '{key}' to {dtype.__name__}.")
                return None
        elif key in ['gender']:
            final_data[key] = 'Male'
        elif key in ['familyhistory']:
            final_data[key] = 'No'
            
    return final_data

def generate_prediction_input(raw_data, scaler):
    gender_val = 2.0 if raw_data.get('gender', 'Male').lower() == 'female' else 1.0
    family_hist_val = 1.0 if raw_data.get('familyhistory', 'No').lower() == 'yes' else 2.0
    
    raw_inputs_for_model = {
        'Age': raw_data.get('age', 30.0),
        'BMI': raw_data.get('bmi', 25.0),
        'SystolicBP': raw_data.get('systolicBP', 120.0),
        'DiastolicBP': raw_data.get('diastolicBP', 80.0),
        'A1c': raw_data.get('hbA1c', 5.5),
        'Glucose': raw_data.get('fastingBloodSugar', 100.0),
        'TotalCholesterol': raw_data.get('cholesterolTotal', 200.0),
        'Triglycerides': raw_data.get('cholesterolTriglycerides', 150.0),
        'Gender': gender_val, 
        'FamilyHistory': family_hist_val
    }

    input_df = pd.DataFrame([raw_inputs_for_model])
    input_df['Gender_2.0'] = 1.0 if raw_inputs_for_model['Gender'] == 2.0 else 0.0
    input_df['FamilyHistory_2.0'] = 1.0 if raw_inputs_for_model['FamilyHistory'] == 2.0 else 0.0
    input_df = input_df.drop(columns=['Gender', 'FamilyHistory'])

    final_input_aligned = pd.DataFrame(0.0, index=[0], columns=FINAL_MODEL_FEATURES, dtype='float64')
    
    for col in input_df.columns:
        if col in final_input_aligned.columns:
            final_input_aligned[col] = input_df[col].iloc[0]

    final_input_aligned = final_input_aligned.reindex(columns=FINAL_MODEL_FEATURES)
    scaled_features = scaler.transform(final_input_aligned[SCALING_COLUMNS])
    final_input_aligned.loc[:, SCALING_COLUMNS] = scaled_features
    
    return final_input_aligned

def predict_diabetes_risk(aligned_data, model):
    prediction_proba = model.predict_proba(aligned_data)[0]
    positive_class_index = list(model.classes_).index(1) 
    probability_of_diabetes = prediction_proba[positive_class_index]
    prediction_label = 1 if probability_of_diabetes >= OPTIMAL_THRESHOLD else 0
    
    return probability_of_diabetes, prediction_label

def identify_diabetes_risk_factors(raw_data, prediction):
    analysis = []
    
    s_bp = raw_data.get('systolicBP', 120.0)
    d_bp = raw_data.get('diastolicBP', 80.0)
    htn_flag = raw_data.get('htn', 0)
    
    if s_bp >= 140 or d_bp >= 90 or htn_flag == 1:
        is_abnormal = True
        bp_status = "High Risk"
        bp_detail = f"Blood pressure is elevated ({s_bp:.1f}/{d_bp:.1f} mmHg). Normal range: <120/<80 mmHg"
        if htn_flag == 1:
            bp_detail += " (Reported history of hypertension)"
        severity = 'high'
    elif s_bp >= 130 or d_bp >= 80:
        is_abnormal = True
        bp_status = "Borderline"
        bp_detail = f"Blood pressure is elevated ({s_bp:.1f}/{d_bp:.1f} mmHg). Target: <120/<80 mmHg"
        severity = 'moderate'
    else:
        is_abnormal = False
        bp_status = "Normal"
        bp_detail = f"Blood pressure is normal ({s_bp:.1f}/{d_bp:.1f} mmHg). Normal range: <120/<80 mmHg"
        severity = 'low'
    
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        analysis.append({
            'title': 'Blood Pressure', 
            'status': bp_status, 
            'detail': bp_detail,
            'severity': severity
        })
    
    a1c = raw_data.get('hbA1c', 5.5)
    glucose = raw_data.get('fastingBloodSugar', 100.0)
    dm_flag = raw_data.get('dm', 0)
    
    if a1c >= 6.5 or glucose >= 126 or dm_flag == 1:
        is_abnormal = True
        dm_status = "High Risk"
        dm_detail = f"Blood sugar levels indicate high risk (FBS: {glucose:.1f} mg/dL, HbA1c: {a1c:.1f}%). Target: FBS <100, HbA1c <5.7%"
        if dm_flag == 1:
            dm_detail += " (Reported history of Diabetes)"
        severity = 'high'
    elif (a1c >= 5.7 and a1c < 6.5) or (glucose >= 100 and glucose < 126):
        is_abnormal = True
        dm_status = "Prediabetes"
        dm_detail = f"Blood sugar levels indicate prediabetes (FBS: {glucose:.1f} mg/dL, HbA1c: {a1c:.1f}%). Target: FBS <100, HbA1c <5.7%"
        severity = 'moderate'
    else:
        is_abnormal = False
        dm_status = "Normal"
        dm_detail = f"Blood sugar levels are normal (FBS: {glucose:.1f} mg/dL, HbA1c: {a1c:.1f}%). Target: FBS <100, HbA1c <5.7%"
        severity = 'low'
    
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        analysis.append({
            'title': 'Blood Sugar Levels', 
            'status': dm_status, 
            'detail': dm_detail,
            'severity': severity
        })

    total_chol = raw_data.get('cholesterolTotal', 200.0)
    trig = raw_data.get('cholesterolTriglycerides', 150.0)

    if total_chol > 240 or trig > 200:
        is_abnormal = True
        lipids_status = "High Risk"
        lipids_detail = f"Cholesterol levels are elevated (Total: {total_chol:.0f}, Triglycerides: {trig:.0f}). Target: Total <200, Triglycerides <150"
        severity = 'high'
    elif total_chol > 200 or trig > 150:
        is_abnormal = True
        lipids_status = "Borderline High"
        lipids_detail = f"Cholesterol levels are borderline high (Total: {total_chol:.0f}, Triglycerides: {trig:.0f}). Target: Total <200, Triglycerides <150"
        severity = 'moderate'
    else:
        is_abnormal = False
        lipids_status = "Normal"
        lipids_detail = f"Cholesterol levels are within target (Total: {total_chol:.0f}, Triglycerides: {trig:.0f}). Target: Total <200, Triglycerides <150"
        severity = 'low'
    
    if (prediction == 1 and is_abnormal) or (prediction == 0 and not is_abnormal):
        analysis.append({
            'title': 'Lipids/Cholesterol', 
            'status': lipids_status, 
            'detail': lipids_detail,
            'severity': severity
        })
    
    return analysis

# ==================== HEART ATTACK FUNCTIONS ====================
def parse_heart_report(file_content):
    data = {}
    lines = file_content.decode('utf-8').strip().split('\n')
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            try:
                if '.' in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value
    
    return data

def create_features_from_data(data):
    df = pd.DataFrame([data])
    
    df['Age_Squared'] = df['Age'] ** 2
    df['BP_Product'] = df['Systolic_BP'] * df['Diastolic_BP']
    df['Chol_Age'] = df['Cholesterol'] * df['Age'] / 100
    df['BMI_Age'] = df['BMI'] * df['Age'] / 100
    df['Risk_Interaction'] = df['CV_Risk_Score'] * df['Metabolic_Score']
    
    return df

def predict_heart_attack_risk(patient_data, model_bundle):
    try:
        model = model_bundle['model']
        scaler = model_bundle['scaler']
        
        df_features = create_features_from_data(patient_data)
        
        X_scaled = scaler.transform(df_features)
        
        prediction_class = model.predict(X_scaled)[0]
        risk_probability = model.predict_proba(X_scaled)[0][1]
    
        return risk_probability, prediction_class
    except Exception as e:
        st.error(f"Error during Heart Attack prediction: {str(e)}")
        return 0.0, 0

def identify_heart_risk_factors(patient_data):
    risk_factors = []
    sbp = patient_data.get('Systolic_BP', 120)
    dbp = patient_data.get('Diastolic_BP', 80)
    if sbp >= 140 or dbp >= 90:
        risk_factors.append({
            'title': 'Hypertension',
            'detail': f"Blood pressure is high ({sbp:.0f}/{dbp:.0f} mmHg). Normal: <120/80",
            'severity': 'high'
        })
    elif sbp >= 130 or dbp >= 85:
        risk_factors.append({
            'title': 'Elevated Blood Pressure',
            'detail': f"Blood pressure is slightly elevated ({sbp:.0f}/{dbp:.0f} mmHg).",
            'severity': 'moderate'
        })
    else:
         risk_factors.append({
            'title': 'Blood Pressure',
            'detail': f"Blood pressure is within normal range ({sbp:.0f}/{dbp:.0f} mmHg).",
            'severity': 'low'
        })

    chol = patient_data.get('Cholesterol', 200)
    if chol >= 240:
        risk_factors.append({
            'title': 'High Cholesterol',
            'detail': f"Total cholesterol is high ({chol:.0f} mg/dL). Target: <200",
            'severity': 'high'
        })
    elif chol >= 200:
         risk_factors.append({
            'title': 'Borderline Cholesterol',
            'detail': f"Total cholesterol is borderline ({chol:.0f} mg/dL). Target: <200",
            'severity': 'moderate'
        })

    bmi = patient_data.get('BMI', 25)
    if bmi >= 30:
        risk_factors.append({
            'title': 'Obesity',
            'detail': f"BMI indicates obesity ({bmi:.1f}). Target: 18.5-24.9",
            'severity': 'high'
        })
    elif bmi >= 25:
        risk_factors.append({
            'title': 'Overweight',
            'detail': f"BMI indicates overweight ({bmi:.1f}). Target: 18.5-24.9",
            'severity': 'moderate'
        })

    trig = patient_data.get('Triglycerides', 150)
    if trig >= 200:
        risk_factors.append({
            'title': 'High Triglycerides',
            'detail': f"Triglycerides are high ({trig:.0f} mg/dL). Target: <150",
            'severity': 'moderate'
        })

    if patient_data.get('Smoking', 0) == 1:
        risk_factors.append({
            'title': 'Smoking History',
            'detail': "Patient has a history of smoking, which significantly increases CV risk.",
            'severity': 'high'
        })
    
    if patient_data.get('Diabetes', 0) == 1:
        risk_factors.append({
            'title': 'Diabetes History',
            'detail': "Patient has diabetes, a major risk factor for heart disease.",
            'severity': 'high'
        })

    if patient_data.get('Stress_Level', 0) >= 7:
        risk_factors.append({
            'title': 'High Stress',
            'detail': f"Reported stress level is high ({patient_data['Stress_Level']}/10).",
            'severity': 'moderate'
        })
    
    return risk_factors

# ==================== DISPLAY FUNCTIONS ====================
def display_results(disease_type, risk_score, prediction, risk_factors, patient_data):
    st.markdown("---")
    
    # Risk Assessment Box
    disease_name = "CKD" if disease_type == "Kidney Disease" else "Diabetes"
    
    if prediction == 0:
        st.markdown(f"""
        <div class="risk-box low-risk">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 2rem; margin-right: 15px;">✅</span>
                <span class="risk-title">Low Risk</span>
            </div>
            <div class="risk-probability">{disease_name} Probability: {risk_score*100:.1f}%</div>
            <p style="margin-top: 15px; color: #155724;">
                Based on the report, there is a <strong>{risk_score*100:.1f}%</strong> probability of the patient having <strong>{disease_type}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-box high-risk">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 2rem; margin-right: 15px;">⚠️</span>
                <span class="risk-title">High Risk</span>
            </div>
            <div class="risk-probability">{disease_name} Probability: {risk_score*100:.1f}%</div>
            <p style="margin-top: 15px; color: #721c24;">
                Based on the report, there is a <strong>{risk_score*100:.1f}%</strong> probability of the patient having <strong>{disease_type}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Factors Section
    st.markdown("###  Identified Risk Factors")
    
    has_significant_risk = any(f['severity'] in ['high', 'moderate'] for f in risk_factors)
    
    if has_significant_risk:
        for factor in risk_factors:
            if factor['severity'] == 'high':
                severity_class = 'risk-factor-red'
                icon = '❤️'
            elif factor['severity'] == 'moderate':
                severity_class = 'risk-factor-yellow'
                icon = '⚠️'
            else:
                severity_class = 'risk-factor-green'
                icon = '✅'
            
            st.markdown(f"""
            <div class="risk-factor {severity_class}">
                <div class="risk-factor-title">{icon} {factor['title']}</div>
                <div class="risk-factor-detail">{factor['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success(" No significant risk factors identified.")

# ==================== MAIN APPLICATION ====================
def main():
    st.markdown('<div class="main-header"> Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error(" No models could be loaded. Please ensure model files are in the correct directory.")
        return
    
    # Disease selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        disease_type = st.selectbox(
            "Select Prediction Type",
            ["Kidney Disease", "Diabetes", "Heart Disease"],
            key="disease_selector"
        )
    
    # File upload
    st.subheader(" Upload Patient Report")
    uploaded_file = st.file_uploader("Choose a report file (.txt)", type=['txt'])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode('utf-8')
        
        # Display file content in table format
        st.subheader(" Report Content")
        
        # Parse and display as table
        lines = file_content.strip().split('\n')
        table_data = []
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                table_data.append({'Parameter': key, 'Value': value})
        
        if table_data:
            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.text_area("Raw Content", file_content, height=200)
        
        # Analyze button
        if st.button(" Analyze Report", type="primary"):
            with st.spinner("Analyzing patient data..."):
                try:
                    if disease_type == "Kidney Disease":
                        if 'ckd' not in models:
                            st.error(" CKD model not loaded!")
                            return
                        
                        patient_data = parse_ckd_report(file_content)
                        risk_score, prediction, data_df = predict_ckd_risk(patient_data, models['ckd'])
                        
                        if risk_score is not None:
                            risk_factors = identify_ckd_risk_factors(patient_data, prediction)
                            display_results(disease_type, risk_score, prediction, risk_factors, patient_data)
                            
                            # Detailed metrics
                            st.markdown("---")
                            st.markdown("### Detailed Patient Metrics")
                            
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
                    
                    if disease_type == "Diabetes":
                        if 'diabetes' not in models:
                            st.error(" Diabetes model not loaded!")
                            return
                        
                        patient_data = parse_diabetes_report(file_content)
                        
                        if patient_data:
                            final_input_df = generate_prediction_input(patient_data, models['diabetes']['scaler'])
                            risk_score, prediction = predict_diabetes_risk(final_input_df, models['diabetes']['model'])
                            risk_factors = identify_diabetes_risk_factors(patient_data, prediction)
                            
                            display_results(disease_type, risk_score, prediction, risk_factors, patient_data)
                            
                            # Detailed metrics
                            st.markdown("---")
                            st.markdown("###  Detailed Patient Metrics")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("Age", f"{patient_data.get('age', 0):.0f} years")
                                st.metric("Systolic BP", f"{patient_data.get('systolicBP', 0):.0f} mmHg")
                                st.metric("BMI", f"{patient_data.get('bmi', 0):.1f}")
                            
                            with metric_col2:
                                st.metric("Glucose", f"{patient_data.get('fastingBloodSugar', 0):.0f} mg/dL")
                                st.metric("Diastolic BP", f"{patient_data.get('diastolicBP', 0):.0f} mmHg")
                                st.metric("HbA1c", f"{patient_data.get('hbA1c', 0):.1f}%")
                            
                            with metric_col3:
                                st.metric("Total Chol", f"{patient_data.get('cholesterolTotal', 0):.0f} mg/dL")
                                st.metric("Triglycerides", f"{patient_data.get('cholesterolTriglycerides', 0):.0f} mg/dL")
                                st.metric("Gender", patient_data.get('gender', 'N/A'))
                            
                            with metric_col4:
                                st.metric("Family History", patient_data.get('familyhistory', 'N/A'))
                                if patient_data.get('htn', 0) == 1:
                                    st.metric("HTN History", "Yes")
                                if patient_data.get('dm', 0) == 1:
                                    st.metric("DM History", "Yes")
                        else:
                            st.error(" Failed to parse the report file.")

                    if disease_type == "Heart Disease":
                        if 'heart' not in models:
                            st.error(" Heart Disease model not loaded!")
                            return
                        
                        patient_data = parse_heart_report(file_content)
                        
                        if patient_data:
                            # Call the prediction function with the model bundle
                            risk_score, prediction = predict_heart_attack_risk(patient_data, models['heart'])
                            
                            # Identify risk factors
                            risk_factors = identify_heart_risk_factors(patient_data)
                            
                            display_results(disease_type, risk_score, prediction, risk_factors, patient_data)
                            
                            # Detailed metrics display
                            st.markdown("---")
                            st.markdown("###  Detailed Patient Metrics")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
                            with metric_col1:
                                st.metric("Age", f"{patient_data.get('Age',0):.0f} years")
                                st.metric("Systolic BP", f"{patient_data.get('Systolic_BP',0):.0f} mmHg")
                                st.metric("Cholesterol", f"{patient_data.get('Cholesterol',0):.0f} mg/dL")
                            
                            with metric_col2:
                                st.metric("Heart Rate", f"{patient_data.get('Heart_Rate',0):.0f} bpm")
                                st.metric("Diastolic BP", f"{patient_data.get('Diastolic_BP',0):.0f} mmHg")
                                st.metric("Triglycerides", f"{patient_data.get('Triglycerides',0):.0f} mg/dL")
                            
                            with metric_col3:
                                st.metric("BMI", f"{patient_data.get('BMI',0):.1f}")
                                st.metric("Exercise", f"{patient_data.get('Exercise_Hours_Per_Week',0):.1f} hrs/week")
                                st.metric("Sleep", f"{patient_data.get('Sleep_Hours_Per_Day',0):.1f} hrs/day")
                            
                            with metric_col4:
                                diabetes_status = "Yes" if patient_data.get('Diabetes',0) == 1 else "No"
                                smoking_status = "Yes" if patient_data.get('Smoking',0) == 1 else "No"
                                st.metric("Diabetes", diabetes_status)
                                st.metric("Smoking", smoking_status)
                                st.metric("Stress Level", f"{patient_data.get('Stress_Level',0):.0f}/10")
                        else:
                            st.error(" Failed to parse the report file.")
                except Exception as e:
                    st.error(f" Error during analysis: {str(e)}")
                    st.exception(e)
    else:
        st.info(" Please upload a patient report file to begin analysis")

if __name__ == "__main__":
    main()