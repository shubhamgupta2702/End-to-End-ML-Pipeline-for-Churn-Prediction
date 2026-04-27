# Telecom Customer Churn Prediction — End-to-End MLOps Pipeline

## Overview

This project is a production-oriented **end-to-end machine learning pipeline** for predicting telecom customer churn. It covers the complete ML lifecycle—from data ingestion and preprocessing to model training, experiment tracking, API serving, containerization, and cloud deployment.

The goal is to identify customers at high risk of churn so businesses can take proactive retention actions.

This repository is designed to demonstrate practical machine learning engineering skills, MLOps practices, and deployment readiness.

---

## Business Problem

Customer churn directly impacts revenue in subscription-based businesses such as telecom.

Predicting churn early helps:

* Reduce customer acquisition costs
* Improve retention strategies
* Increase customer lifetime value
* Prioritize intervention campaigns

This model predicts whether a customer is likely to churn based on service usage, contract type, billing behavior, and support history.

---

## Key Highlights

* Built a complete **end-to-end ML pipeline**
* Production-ready modular architecture
* Automated preprocessing and training workflows
* Hyperparameter tuning with GridSearchCV
* MLflow experiment tracking integration
* FastAPI-based inference API
* Streamlit frontend for interactive prediction
* Dockerized application for reproducibility
* CI/CD automation using GitHub Actions
* AWS deployment (EC2 / Elastic Beanstalk ready)
* Health checks and logging implemented
* Latency monitoring for inference performance

---

## Model Performance

Dataset Size: **7,000+ customer records**

Primary objective was to optimize recall for churn detection.

### Metrics

| Metric              |   Score |
| ------------------- | ------: |
| Recall              |     79% |
| F1 Score            |  62–63% |
| Average API Latency |  ~21 ms |
| p95 Latency         | < 30 ms |

Why recall matters:
Missing churners is costlier than false positives in retention workflows.

---

## Tech Stack

### Machine Learning

* Scikit-learn
* XGBoost
* CatBoost
* AdaBoost
* Gradient Boosting
* Logistic Regression

### Backend

* FastAPI
* Pydantic

### Frontend

* Streamlit

### MLOps

* Docker
* GitHub Actions (CI/CD)
* MLflow
* AWS EC2 / Elastic Beanstalk

### Data

* Pandas
* NumPy

---

## Project Architecture

```text
├── artifacts/              # Trained models and preprocessors
├── notebook/               # EDA and experimentation
├── src/
│   ├── components/         # Training pipeline components
│   ├── pipeline/           # Training and prediction pipelines
│   ├── utils/              # Utility functions
│   ├── logger/             # Logging configuration
│   ├── exception/          # Custom exception handling
├── app.py                  # FastAPI application
├── Dockerfile              # Container configuration
├── requirements.txt        # Dependencies
├── .github/workflows/      # CI/CD workflows
├── .ebextensions/          # Elastic Beanstalk configs
└── README.md
```

---

## ML Pipeline Flow

1. Data ingestion
2. Data transformation
3. Feature engineering
4. Model training
5. Hyperparameter tuning
6. Model evaluation
7. Model persistence
8. Experiment tracking
9. API serving
10. Deployment

---

## Experiment Tracking (MLflow)

MLflow is integrated for:

* Parameter logging
* Metric logging
* Model artifact tracking
* Experiment reproducibility

Tracked metrics include:

* Precision
* Recall
* F1 Score
* ROC-AUC

---

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Welcome to Churn Prediction API Model."
}
```

### Prediction Endpoint

```http
POST /predict
```

Sample request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.50,
  "TotalCharges": 550.25
}
```

Sample response:

```json
{
  "prediction": "Yes",
  "probability": 0.83,
  "latency_ms": 19.81
}
```

Swagger docs:

```text
http://localhost:8000/docs
```

---

## Running Locally

### Clone repository

```bash
git clone <https://github.com/shubhamgupta2702/End-to-End-ML-Pipeline-for-Churn-Prediction.git>
cd <repository-folder>
```

### Create environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
myvenv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run training pipeline

```bash
python src/pipeline/training_pipeline.py
```

### Run FastAPI server

```bash
uvicorn app:app --reload
```

---

## Docker Setup

### Build image

```bash
docker build -t churn-api .
```

### Run container

```bash
docker run -p 8000:8000 churn-api
```

---

## CI/CD Pipeline

GitHub Actions workflow automates:

* Dependency installation
* Validation checks
* Docker build
* Deployment to EC2

Workflow file:

```text
.github/workflows/main.yml
```

---

## Cloud Deployment

Deployment-ready for:

* AWS EC2
* AWS Elastic Beanstalk

Production features:

* Dockerized deployment
* Health checks
* Logging
* Environment configuration

---

## Engineering Decisions

### Why FastAPI?

High-performance async API framework with automatic docs.

### Why Docker?

Environment consistency and deployment portability.

### Why MLflow?

Experiment reproducibility and model lifecycle management.

### Why Recall Optimization?

In churn prediction, missing churners has a higher business cost.

### Shubham Gupta ->>

Built as a production-grade machine learning engineering project to demonstrate end-to-end ML system design, deployment, and MLOps capabilities.
