from fastapi import FastAPI, HTTPException
from model.predict import predict
from app.schema import PredictionRequest

app = FastAPI(title="ML API")

@app.get("/")
def root():
    return {"message": "API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(request: PredictionRequest):
    try:
        result = predict(request)
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_proba")
def get_prediction_proba(request: PredictionRequest):
    try:
        result = predict(request, proba=True)
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))