import pandas as pd

from src.train_model import (
    preprocess_data,
    split_data,
    train_model,
    train_vectorizer,
)


def test_model_training():

    df = pd.DataFrame(
        {
            "Summary": [
                # High (5)
                "Application crashes after login",
                "Application closes unexpectedly",
                "System crashes while saving data",
                "Payment page crashes on submit",
                "Frequent application crash on startup",
                # Medium (5)
                "Unable to login",
                "Login authentication failed",
                "Password reset email not received",
                "User profile update not working",
                "Session expires too quickly",
                # Low (5)
                "Button color is wrong",
                "UI alignment issue",
                "Typo in dashboard title",
                "Logo slightly misaligned",
                "Incorrect icon displayed on settings page",
                # Highest (5)
                "Database connection timeout",
                "Database server down",
                "Production API unavailable",
                "Critical security vulnerability detected",
                "Payment service completely unavailable",
            ],
            "Priority": [
                # High
                "High",
                "High",
                "High",
                "High",
                "High",
                # Medium
                "Medium",
                "Medium",
                "Medium",
                "Medium",
                "Medium",
                # Low
                "Low",
                "Low",
                "Low",
                "Low",
                "Low",
                # Highest
                "Highest",
                "Highest",
                "Highest",
                "Highest",
                "Highest",
            ],
        }
    )

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test, encoder = split_data(df)

    X_train, X_test, vectorizer = train_vectorizer(
        X_train,
        X_test,
    )

    model = train_model(
        X_train,
        y_train,
    )

    assert model is not None
