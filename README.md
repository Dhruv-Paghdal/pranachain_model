# PranaChain AI-Powered Predictive Health Platform

This application serves as a proof-of-concept for PranaChain's Industry-Academic Collaboration initiative. It showcases the artificial intelligence aspect of the company's "AI + Blockchain Synergy" strategy through an intuitive web interface that executes advanced machine learning algorithms for disease risk assessment. Built using Python and **Streamlit**, the platform incorporates three distinct **Scikit-learn** models designed to evaluate the likelihood of Diabetes, Heart Disease, and Chronic Kidney Disease.

## Key Capabilities

* **Triple-Model Integration:** Incorporates prediction algorithms for three major health conditions.
* **Intuitive User Interface:** A streamlined, easy-to-navigate design powered by Streamlit, enabling users without technical expertise to generate predictions.
* **Text File Upload System:** Users can submit patient information via text file (`.txt` format) and receive immediate probability assessments.
* **Unified Platform:** All three prediction models are housed within a single, cohesive web application.

## Model Accuracy Metrics

Each model was developed using the Random Forest algorithm and demonstrates excellent performance characteristics:

| Prediction Model | Accuracy Rate | Precision Score | Recall Score | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Kidney Disease** | 98.9% | 98.0% | 99.7% | 98.8% |
| **Diabetes** | 98.6% | 99.0% | 99.0% | 99.0% |
| **Heart Disease** | 92.0% | 92.0% | 92.0% | 92.0% |

## Technology Stack

* **Core Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning & Analytics:**
    * Scikit-learn (model development and execution)
    * Pandas (data processing)
    * NumPy
* **Model Storage:** Joblib

## Installation Guide

Follow the instructions below to configure and launch the application locally.

**Required Software:**

* [Python 3.x](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)

**Installation Steps:**

1. **Clone the GitHub repository:**
   ```bash
   https://github.com/Dhruv-Paghdal/pranachain_model.git
   ```

2. **Change to the project folder:**
   ```bash
   cd diseasePredictionModels
   ```

3. **Install project dependencies:**
   The complete list of required packages is available in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
   *Important: Pre-trained model files (`.joblib` format) are already included in the `models/` folder.*

## Launching the Application

1. Verify you are in the main project directory (containing `app.py`).

2. Execute the following command in your terminal:
   ```bash
   streamlit run app.py
   ```

3. Streamlit will launch automatically in your default browser, typically at `http://localhost:8501`.

## Application Usage Instructions

1. After launching the app, navigate to the sidebar and choose your desired prediction model:
    * Diabetes
    * Heart Disease
    * Kidney Disease

2. On the selected model's page, use the "Browse files" option to upload a text file (`.txt` format) with patient information.

3. **Sample test files** are available in the `test_file/` folder for demonstration purposes.

4. The system will analyze the uploaded data and present the prediction results directly on the screen.

## Directory Organization

```
.
├── app.py                # Primary Streamlit application file
├── models/
│   ├── ckd_model_pred.joblib
│   ├── diabetes.joblib
│   └── heart.joblib
├── code/                 # Training scripts and Jupyter Notebooks
│   ├── chronic_kidney_disease/
│   ├── diabetes/
│   └── heart/
├── test_file/            # Sample .txt files for demonstration
│   ├── diabetes_report_1.txt
│   ├── heart_report_1.txt
│   └── kidney_report_1.txt
├── requirements.txt      # Python package dependencies
└── README.md             # Documentation file
```
