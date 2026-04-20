from fastapi import FastAPI, HTTPException
from app.schema import PredictionRequest
from model.predict import predict

app = FastAPI(title="ML API")

@app.get("/")
def root():
    return {"message": "API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(request: dict):
    try:
        result = predict(request)
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))