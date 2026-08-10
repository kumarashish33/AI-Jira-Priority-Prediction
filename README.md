# AI-Powered Jira Ticket Priority Prediction

Software Engineering for Machine Learning (SE4ML)

Version 1.0

---

## Project Overview

This project predicts the priority of Jira tickets using Natural Language Processing (NLP) and Machine Learning.

The application preprocesses Jira ticket summaries, converts them into TF-IDF vectors, and predicts the ticket priority using a Logistic Regression model.

---

## Features

- NLP preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Streamlit Web Application
- FastAPI REST API
- Logging
- Exception Handling
- Automated Testing using pytest
- Data Quality Validation

---

## Project Structure

```text
AI-Jira-Priority-Prediction/
│
├── api/
├── artifacts/
├── data/
├── logs/
├── notebooks/
├── src/
├── tests/
├── streamlit_app.py
└── requirements.txt
```

---

## Technology Stack

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- FastAPI
- Pytest
- Joblib

---

## Model Pipeline

Dataset

↓

Text Cleaning

↓

TF-IDF

↓

Logistic Regression

↓

Prediction

---

## Performance

| Metric | Value |
|---------|------:|
| Accuracy | 99.xx% |
| Macro F1 | 0.99xx |

---

## Running the Project

### Install

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python -m src.train_model
```

### Streamlit

```bash
streamlit run streamlit_app.py
```

### FastAPI

```bash
uvicorn api.app:app --reload
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

## Running Tests

```bash
pytest
```

---

## Data Quality Checks

- Schema Validation
- Missing Value Check

---

## Future Improvements

- SVM
- XGBoost
- Cross Validation
- Model Monitoring
