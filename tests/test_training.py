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
                "Application crashes after login",
                "Application closes unexpectedly",
                "Unable to login",
                "Login authentication failed",
                "Button color is wrong",
                "UI alignment issue",
                "Database connection timeout",
                "Database server down",
            ],
            "Priority": [
                "High",
                "High",
                "Medium",
                "Medium",
                "Low",
                "Low",
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
