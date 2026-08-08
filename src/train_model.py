import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_text, initialize_nltk

from src.config import (
    DATA_DIR,
    MODEL_PATH,
    TFIDF_PATH,
    LABEL_ENCODER_PATH,
    RAW_DATA_PATH,
)

from src.preprocessing import clean_text
from src.logging_config import logger


def load_data():
    logger.info("Loading dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df

def preprocess_data(df):
    logger.info("Starting preprocessing...")

    df = df.copy()

    df = df.dropna(subset=["Summary", "Priority"])

    df["Clean_Summary"] = df["Summary"].apply(clean_text)

    logger.info("Preprocessing completed.")

    return df

def split_data(df):
    logger.info("Splitting dataset...")

    X = df["Clean_Summary"]

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["Priority"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, label_encoder

def train_vectorizer(X_train, X_test):
    logger.info("Training TF-IDF vectorizer...")

    tfidf = TfidfVectorizer()

    X_train = tfidf.fit_transform(X_train)

    X_test = tfidf.transform(X_test)

    return X_train, X_test, tfidf

def train_model(X_train, y_train):
    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed.")

    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    logger.info("=" * 50)
    logger.info(f"Accuracy      : {accuracy:.4f}")
    logger.info(f"Macro F1 Score: {macro_f1:.4f}")
    logger.info("=" * 50)

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    return accuracy, macro_f1

def save_artifacts(model, vectorizer, encoder):
    logger.info("Saving artifacts...")

    joblib.dump(model, MODEL_PATH)

    joblib.dump(vectorizer, TFIDF_PATH)

    joblib.dump(encoder, LABEL_ENCODER_PATH)

    logger.info("Artifacts saved successfully.")

def main():
    try:
        initialize_nltk()
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

        evaluate_model(
            model,
            X_test,
            y_test,
        )

        save_artifacts(
            model,
            vectorizer,
            encoder,
        )

        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()