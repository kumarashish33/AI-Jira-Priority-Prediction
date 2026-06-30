import joblib

from .preprocessing import clean_text
from .config import MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH
from .logging_config import logger

logger.info("Loading model...")
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
logger.info("Model loaded successfully.")

def predict_priority(ticket):
    logger.info('Prediction started for...')
    logger.info(ticket)
    if not isinstance(ticket,str):
        logger.error("Recieved ticket summary is not string.")
        raise TypeError('Sample should be string.')
    
    if ticket.strip() == "":
        logger.error("Empty ticket summary received.")
        raise ValueError('Ticket summary cannot be empty.')

    cleaned_text = clean_text(ticket)

    ticket_vector = tfidf.transform([cleaned_text])

    prediction = model.predict(ticket_vector)

    probabilities = model.predict_proba(ticket_vector)
    class_probabilities = dict(
        zip(
            label_encoder.classes_,
            probabilities[0]
        )
    )
    confidence = probabilities.max()

    priority = label_encoder.inverse_transform(prediction)

    logger.info(f"Predicted Priority: {priority[0]}")

    return priority[0],confidence,class_probabilities

