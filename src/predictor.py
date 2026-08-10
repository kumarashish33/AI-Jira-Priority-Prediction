import joblib

from .config import LABEL_ENCODER_PATH, MODEL_PATH, TFIDF_PATH
from .logging_config import logger
from .preprocessing import clean_text

# Load artifacts safely
try:
    logger.info("Loading model artifacts...")

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    logger.info("Model artifacts loaded successfully.")

except Exception:
    logger.exception("Failed to load model artifacts.")
    raise


def predict_priority(ticket):
    try:
        logger.info("Prediction request received.")

        if not isinstance(ticket, str):
            logger.error("Input is not a string.")
            raise TypeError("Ticket summary must be a string.")

        ticket = ticket.strip()

        if not ticket:
            logger.error("Empty ticket summary received.")
            raise ValueError("Ticket summary cannot be empty.")

        cleaned_text = clean_text(ticket)

        ticket_vector = tfidf.transform([cleaned_text])

        prediction = model.predict(ticket_vector)

        probabilities = model.predict_proba(ticket_vector)

        confidence = float(probabilities.max())

        class_probabilities = dict(
            zip(
                label_encoder.classes_,
                probabilities[0],
            )
        )

        priority = label_encoder.inverse_transform(prediction)[0]

        logger.info(
            f"Prediction completed successfully. "
            f"Priority={priority}, Confidence={confidence:.4f}"
        )

        return priority, confidence, class_probabilities

    except Exception:
        logger.exception("Prediction failed.")
        raise
