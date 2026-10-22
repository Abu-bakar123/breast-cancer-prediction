# Breast Cancer Prediction System

## Overview
An educational machine learning project that predicts a breast cancer class using a custom Logistic Regression model implemented with NumPy. The application uses FastAPI as the backend API and Streamlit as the interactive frontend.

## Technologies
- Python
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Requests
- Joblib

## Project Structure
```text
cancer-model/
├── app.py
├── custom_model.py
├── streamlit_app.py
├── cancer_model_params.json
├── cancer_scaler.joblib
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Start the FastAPI backend
```bash
python -m uvicorn app:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### 3. Start the Streamlit frontend
Open a second terminal and run:
```bash
python -m streamlit run streamlit_app.py
```

## Deployment
- Source code: GitHub
- Backend API: Render
- Frontend: Streamlit Community Cloud

Configure the Streamlit frontend to send requests to the deployed Render API URL.

## Disclaimer
This project is for educational purposes only. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.