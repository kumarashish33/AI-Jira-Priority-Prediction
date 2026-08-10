from src.logging_config import logger

REQUIRED_COLUMNS = [
    "Summary",
    "Priority",
]


def check_missing_values(df):
    missing = df.isnull().mean().mul(100).round(2)

    logger.info("\nMissing Value (%)")
    logger.info(missing)

    return missing


def validate_schema(df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    logger.info("Dataset schema validation passed.")
