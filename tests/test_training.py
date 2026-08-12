from src.train_model import (load_data, preprocess_data, split_data,
                             train_model, train_vectorizer)


def test_model_training():
    df = load_data()

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
