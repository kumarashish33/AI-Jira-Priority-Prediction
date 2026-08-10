from fastapi import FastAPI, HTTPException

from src.predictor import predict_priority

from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="AI Jira Priority Prediction API",
    version="1.0.0",
    description="REST API for Jira ticket priority prediction",
)


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        priority, confidence, probabilities = predict_priority(
            request.text,
        )

        return PredictionResponse(
            priority=priority, confidence=confidence, probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
