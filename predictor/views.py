import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import TemporaryUploadedFile
import pickle
import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ckd_stacking_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        models_dict = pickle.load(f)

    svc_model = models_dict['svc']
    rf_model = models_dict['rf']
    xgb_model = models_dict['xgb']
    meta_model = models_dict['meta']
    scaler_ckd = models_dict['scaler_ckd']
    optimal_threshold = models_dict['optimal_threshold']
    common_features = models_dict['common_features']
    features_m3 = models_dict['features_m3']
    print("CKD Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_input(data):
    input_dict = {
        'age': float(data['age']),
        'SystolicBP': float(data['systolicBP']),
        'DiastolicBP': float(data['diastolicBP']),
        'FastingBloodSugar': float(data['fastingBloodSugar']),
        'HbA1c': float(data['hbA1c']),
        'sc': float(data['serumCreatinine']),
        'GFR': float(data['gfr']),
        'hemo': float(data['hemoglobin']),
        'CholesterolTotal': float(data['cholesterolTotal']),
        'CholesterolLDL': float(data['cholesterolLDL']),
        'CholesterolHDL': float(data['cholesterolHDL']),
        'CholesterolTriglycerides': float(data['cholesterolTriglycerides']),
        'al': float(data.get('proteinInUrine', 0)),
        'ACR': float(data['acr'])
    }
    
    input_dict['htn'] = 1 if (input_dict['SystolicBP'] >= 140 or input_dict['DiastolicBP'] >= 90) else 0
    input_dict['dm'] = 1 if (input_dict['FastingBloodSugar'] >= 126 or input_dict['HbA1c'] >= 6.5) else 0
    
    df = pd.DataFrame([input_dict])
    
    X_full = df[features_m3]
    X_common = df[common_features]
    
    binary_cols = ['htn', 'dm']
    cols_to_scale = [col for col in X_full.columns if col not in binary_cols and X_full[col].dtype in [np.float64, np.int64]]
    
    X_full_scaled = X_full.copy()
    X_full_scaled[cols_to_scale] = scaler_ckd.transform(X_full[cols_to_scale])
    
    X_common_scaled = X_full_scaled[common_features]
    
    return X_full_scaled, X_common_scaled, input_dict

def analyze_risk_factors(data):
    risks = []
    
    systolic = float(data['SystolicBP'])
    diastolic = float(data['DiastolicBP'])
    if systolic >= 140 or diastolic >= 90:
        severity = 'high' if (systolic >= 160 or diastolic >= 100) else 'moderate'
        risks.append({
            'category': 'Hypertension',
            'severity': severity,
            'message': f'Blood pressure is elevated ({systolic}/{diastolic} mmHg). Normal range: <120/80 mmHg'
        })
    
    fbs = float(data['FastingBloodSugar'])
    hba1c = float(data['HbA1c'])
    if fbs >= 126 or hba1c >= 6.5:
        severity = 'high' if (fbs >= 160 or hba1c >= 8) else 'moderate'
        risks.append({
            'category': 'Diabetes',
            'severity': severity,
            'message': f'Blood sugar levels indicate diabetes (FBS: {fbs} mg/dL, HbA1c: {hba1c}%). Target: FBS <100, HbA1c <5.7%'
        })
    
    sc = float(data['sc'])
    gfr = float(data['GFR'])
    if sc > 1.2 or gfr < 60:
        severity = 'high' if (sc > 2.0 or gfr < 30) else 'moderate'
        risks.append({
            'category': 'Kidney Function',
            'severity': severity,
            'message': f'Kidney function is impaired (Creatinine: {sc} mg/dL, GFR: {gfr}). Normal GFR: >90'
        })
    
    hemo = float(data['hemo'])
    if hemo < 12:
        severity = 'high' if hemo < 10 else 'moderate'
        risks.append({
            'category': 'Anemia',
            'severity': severity,
            'message': f'Hemoglobin is low ({hemo} g/dL). Normal range: 12-16 g/dL'
        })
    
    chol_total = float(data['CholesterolTotal'])
    chol_ldl = float(data['CholesterolLDL'])
    if chol_total > 200 or chol_ldl > 130:
        severity = 'high' if (chol_total > 240 or chol_ldl > 160) else 'moderate'
        risks.append({
            'category': 'High Cholesterol',
            'severity': severity,
            'message': f'Cholesterol levels are elevated (Total: {chol_total}, LDL: {chol_ldl}). Target: Total <200, LDL <100'
        })
    
    acr = float(data['ACR'])
    if acr > 30:
        severity = 'high' if acr > 300 else 'moderate'
        risks.append({
            'category': 'Proteinuria',
            'severity': severity,
            'message': f'Protein in urine is elevated (ACR: {acr} mg/g). Normal: <30 mg/g'
        })
    
    return risks
   
def process_file_input(file):
    expected_fields = [
        'age', 'systolicBP', 'diastolicBP', 'fastingBloodSugar', 'hbA1c',
        'serumCreatinine', 'gfr', 'hemoglobin', 'cholesterolTotal',
        'cholesterolLDL', 'cholesterolHDL', 'cholesterolTriglycerides',
        'proteinInUrine', 'acr'
    ]
    data = {}
    
    try:
        content = file.read().decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error reading file content: {e}")

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key in expected_fields:
                data[key] = value

    missing_fields = [field for field in expected_fields if field not in data or not data[field]]
    if missing_fields:
        raise ValueError(f"Missing or empty fields in the file: {', '.join(missing_fields)}")

    converted_data = {}
    validation_rules = {
        'age': {'type': int, 'min': 1, 'max': 120, 'message': 'Age must be an integer between 1 and 120.'},
        'systolicBP': {'type': float, 'min': 50, 'max': 300, 'message': 'Systolic BP must be a number between 50 and 300.'},
        'diastolicBP': {'type': float, 'min': 30, 'max': 200, 'message': 'Diastolic BP must be a number between 30 and 200.'},
        'fastingBloodSugar': {'type': float, 'min': 50, 'max': 500, 'message': 'FBS must be a number between 50 and 500.'},
        'hbA1c': {'type': float, 'min': 3.0, 'max': 20.0, 'message': 'HbA1c must be a number between 3.0 and 20.0.'},
        'serumCreatinine': {'type': float, 'min': 0.1, 'max': 10.0, 'message': 'Serum Creatinine must be a number between 0.1 and 10.0.'},
        'gfr': {'type': float, 'min': 5.0, 'max': 150.0, 'message': 'GFR must be a number between 5.0 and 150.0.'},
        'hemoglobin': {'type': float, 'min': 5.0, 'max': 20.0, 'message': 'Hemoglobin must be a number between 5.0 and 20.0.'},
        'cholesterolTotal': {'type': float, 'min': 50, 'max': 600, 'message': 'Total Cholesterol must be a number between 50 and 600.'},
        'cholesterolLDL': {'type': float, 'min': 10, 'max': 400, 'message': 'LDL Cholesterol must be a number between 10 and 400.'},
        'cholesterolHDL': {'type': float, 'min': 5, 'max': 150, 'message': 'HDL Cholesterol must be a number between 5 and 150.'},
        'cholesterolTriglycerides': {'type': float, 'min': 10, 'max': 1000, 'message': 'Triglycerides must be a number between 10 and 1000.'},
        'acr': {'type': float, 'min': 0.0, 'max': 2000.0, 'message': 'ACR must be a non-negative number.'},
        'proteinInUrine': {'type': float, 'choices': [0.0, 1.0, 2.0, 3.0, 4.0], 'message': 'Protein in Urine must be a valid selection (0, 1, 2, 3, or 4).'}
    }

    for key, rules in validation_rules.items():
        raw_value = data.get(key, '').strip()
        if not raw_value: continue 

        try:
            value = rules['type'](raw_value)
        except ValueError:
            raise ValueError(f"Invalid data type for {key}: {raw_value}. {rules['message']}")

        if value < 0:
             raise ValueError(f"Negative value detected for {key}: {value}. All inputs must be non-negative.")

        if 'choices' in rules and value not in rules['choices']:
            raise ValueError(f"Invalid value for {key}: {value}. {rules['message']}")
        
        if 'min' in rules and value < rules['min']:
            raise ValueError(f"Value for {key} is too low: {value}. {rules['message']}")

        if 'max' in rules and value > rules['max']:
            raise ValueError(f"Value for {key} is too high: {value}. {rules['message']}")
            
        converted_data[key] = value
    
    if len(converted_data) != len(expected_fields):
        processed_keys = set(converted_data.keys())
        expected_keys = set(expected_fields)
        missing = expected_keys - processed_keys
        if missing:
             raise ValueError(f"Missing required data fields: {', '.join(missing)}.")

    return converted_data  

@csrf_exempt 
def predict_ckd_file(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method is allowed'}, status=405)

    if 'reportFile' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No file uploaded.'}, status=400)

    report_file = request.FILES['reportFile']
    
    if not report_file.name.lower().endswith('.txt'):
        return JsonResponse({'success': False, 'error': 'Invalid file format. Please upload a .txt file.'}, status=400)

    try:
        data = process_file_input(report_file)
        
        X_full_scaled, X_common_scaled, processed_data = preprocess_input(data)
        
        m1_proba = svc_model.predict_proba(X_common_scaled)[:, 1][0]
        m2_proba = rf_model.predict_proba(X_common_scaled)[:, 1][0]
        m3_proba = xgb_model.predict_proba(X_full_scaled)[:, 1][0]
        
        meta_features = pd.DataFrame({
            'M1_Proba': [m1_proba],
            'M2_Proba': [m2_proba],
            'M3_Proba': [m3_proba]
        })
        
        final_proba = meta_model.predict_proba(meta_features)[:, 1][0]
        has_ckd = final_proba >= optimal_threshold
        ckd_probability = final_proba * 100
        
        risks = analyze_risk_factors(processed_data)
                
        response = {
            'success': True,
            'prediction': {
                'hasCKD': bool(has_ckd),
                'probability': round(ckd_probability, 1),
                'confidence': 'High' if abs(final_proba - 0.5) > 0.3 else 'Moderate'
            },
            'insights': {
                'riskLevel': 'High' if ckd_probability > 70 else 'Moderate' if ckd_probability > 40 else 'Low',
                'riskFactors': risks,
            },
            'modelDetails': {
                'baseModelPredictions': {
                    'SVC': round(m1_proba * 100, 1),
                    'RandomForest': round(m2_proba * 100, 1),
                    'XGBoost': round(m3_proba * 100, 1)
                },
                'optimalThreshold': round(optimal_threshold * 100, 1)
            }
        }
        
        return JsonResponse(response)
    
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Prediction error from file upload: {e}", exc_info=True)
        
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred during prediction from the file.'
        }, status=400)  

def ckd_interface(request):
    return render(request, 'predictor/ckd_predictor.html')