from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import boto3
import pickle
import traceback
import os
from datetime import datetime
from dotenv import load_dotenv

# =============================
# LOAD ENV VARIABLES
# =============================
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-gold-price-fjs")
MODEL_KEY = os.getenv("MODEL_KEY", "models/model.pkl")

LAGS = int(os.getenv("LAGS", 5))

# =============================
# LOGGER
# =============================
def log(msg):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}", flush=True)

log("🚀 STARTING FASTAPI S3 MODEL SERVICE (DOTENV MODE)")

# =============================
# S3 CLIENT (WITH DOTENV CREDS)
# =============================
log("☁️ Initializing S3 client with dotenv credentials...")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

log("✅ S3 client initialized")

# =============================
# FASTAPI APP
# =============================
app = FastAPI(
    title="Gold Price Prediction API (S3 Model)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# LOAD MODEL FROM S3
# =============================
def load_model_from_s3():
    try:
        log("📦 Loading model from S3...")
        log(f"Bucket: {BUCKET_NAME}")
        log(f"Key: {MODEL_KEY}")

        obj = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_KEY
        )

        model_bytes = obj["Body"].read()
        log(f"📊 Model size: {len(model_bytes)} bytes")

        model = pickle.loads(model_bytes)

        log("✅ Model loaded successfully from S3")
        return model

    except Exception:
        log("❌ FAILED TO LOAD MODEL FROM S3")
        log(traceback.format_exc())
        raise


# =============================
# SAFE STARTUP LOAD
# =============================
log("⚙️ Attempting model load at startup...")

try:
    model = load_model_from_s3()
    log("🚀 API READY (MODEL LOADED)")
except Exception as e:
    model = None
    log("⚠️ API STARTED WITHOUT MODEL (DEGRADED MODE)")
    log(str(e))


# =============================
# REQUEST MODEL
# =============================
class PredictionRequest(BaseModel):
    prices: list[float]


# =============================
# ROOT
# =============================
@app.get("/")
def root():
    log("🏠 Root endpoint called")
    return {"message": "Gold Price Prediction API running"}


# =============================
# PREDICT ENDPOINT
# =============================
@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        log("📥 NEW PREDICTION REQUEST")
        log(f"Input: {request.prices}")

        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Check S3 connection or credentials."
            )

        input_data = np.array(request.prices)

        log(f"📏 Input shape: {input_data.shape}")

        if len(input_data) != LAGS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {LAGS} values, got {len(input_data)}"
            )

        input_data = input_data.reshape(1, -1)

        log("🤖 Running prediction...")
        prediction = model.predict(input_data)

        result = float(prediction[0])

        log(f"🎯 Prediction result: {result}")

        return {"predicted_price": result}

    except HTTPException as he:
        raise he

    except Exception as e:
        log("❌ ERROR DURING PREDICTION")
        log(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# RUN LOCALLY
# =============================
if __name__ == "__main__":
    import uvicorn

    log("🔥 STARTING UVICORN SERVER")

    uvicorn.run(
        "api_from_s3:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )