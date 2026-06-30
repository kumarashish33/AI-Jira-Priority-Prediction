from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Data folder path
DATA_DIR = PROJECT_ROOT/"data"
# Model folder Path 
ARTIFACTS_DIR = PROJECT_ROOT/"artifacts"

# Model Files
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.pkl"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
