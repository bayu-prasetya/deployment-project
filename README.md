# 🚀 Production-Ready ML API with FastAPI, Docker & Railway

A complete **end-to-end Machine Learning system** that covers:

* Model training pipeline (modular & scalable)
* DataFrame-based preprocessing using `sklearn Pipeline`
* REST API using FastAPI
* Containerization with Docker
* Automated deployment via Railway (CI/CD)

---

# 🌟 Highlights

✅ Production-grade ML pipeline (modular & clean)
✅ Data validation using Pydantic
✅ Consistent preprocessing with `ColumnTransformer`
✅ Dockerized for reproducibility
✅ One-click deployment with Railway

---

# 🧠 Architecture Overview

```text
User → FastAPI → Sklearn Pipeline → Model → Prediction

GitHub → Railway → Docker Build → Live API
```

---

# 📁 Project Structure

```
ml-railway-project/
│
├── app/                # FastAPI application
├── model/              # Model loading & inference
├── pipeline/           # Training workflow (modular)
├── tests/              # API testing
├── Dockerfile          # Container config
├── requirements.txt    # Dependencies
└── railway.json        # Deployment config
```

---

# ⚙️ How It Works

## 1. Training Pipeline

* Load data
* Split dataset
* Train model (with sklearn Pipeline)
* Evaluate performance
* Save model (`model_v1.pkl`)

## 2. API Layer

* Accept request via JSON
* Validate input using Pydantic
* Convert to DataFrame
* Run prediction
* Return result

## 3. Deployment

* Push to GitHub
* Railway auto-build Docker image
* API goes live

---

# 🧪 Run Locally

## 1. Install dependencies

```
pip install -r requirements.txt
```

## 2. Train model

```
python pipeline/pipeline.py
```

## 3. Run API

```
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# 🐳 Run with Docker

```
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

---

# ☁️ Deploy to Railway

1. Push repo to GitHub
2. Go to [https://railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Generate domain

Done ✅

---

# 🔌 API Endpoints

## Health Check

```
GET /health
```

## Prediction

```
POST /predict
```

### Request

```json
{
  "age": 30,
  "income": 8000,
  "gender": "M"
}
```

### Response

```json
{
  "prediction": [1]
}
```

---

# 🧠 Key Design Decisions

* **Sklearn Pipeline** → ensures consistency between training & inference
* **Pydantic** → prevents invalid input data
* **Docker** → reproducible environment
* **Railway** → simple deployment & CI/CD

---

# 🚀 Future Improvements

* Add prediction logging
* Integrate MLflow
* Add authentication
* Deploy to Cloud Run / Kubernetes
* Add monitoring & alerting

---

# 🎤 Interview Talking Points

* Built an end-to-end ML system, not just a model
* Ensured consistency using sklearn Pipeline
* Implemented schema validation to prevent runtime errors
* Used Docker for reproducibility
* Automated deployment with Railway

---

# 🏁 Final Note

This project demonstrates a **real-world ML deployment workflow**, bridging the gap between data science and production engineering.

> Not just building models — but building systems 🚀
