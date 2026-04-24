from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import boto3
import pickle
import traceback
import os
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# =============================
# LOAD ENV VARIABLES
# =============================
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-gold-price-fjs")

# IMPORTANT: FIXED MODEL FILE
MODEL_KEY = os.getenv("MODEL_KEY", "models/model.pkl")

LAGS = int(os.getenv("LAGS", 5))

# =============================
# LOGGER
# =============================
def log(message):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {message}", flush=True)

log("STARTING FASTAPI S3 MODEL SERVICE (SINGLE MODEL RELOAD)")

# =============================
# S3 CLIENT
# =============================
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

log("S3 CLIENT INITIALIZED")

# =============================
# FASTAPI APP
# =============================
app = FastAPI(
    title="Gold Price Prediction API (Single Model Reload)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# GLOBAL MODEL
# =============================
model = None

# =============================
# LOAD MODEL FROM S3
# =============================
def load_model_from_s3():
    global model

    try:
        log("LOADING MODEL FROM S3")

        response = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_KEY
        )

        model_bytes = response["Body"].read()

        log(f"MODEL SIZE: {len(model_bytes)} BYTES")

        loaded_model = pickle.loads(model_bytes)

        model = loaded_model

        log("MODEL LOADED SUCCESSFULLY")

    except Exception:
        log("FAILED TO LOAD MODEL")
        log(traceback.format_exc())

# =============================
# TIME UNTIL MIDNIGHT UTC
# =============================
def seconds_until_midnight_utc():
    now = datetime.utcnow()
    tomorrow = now.date() + timedelta(days=1)
    midnight = datetime.combine(tomorrow, datetime.min.time())
    return (midnight - now).total_seconds()

# =============================
# MIDNIGHT RELOAD THREAD
# =============================
def midnight_model_reloader():
    while True:
        wait_seconds = seconds_until_midnight_utc()
        log(f"SLEEPING {wait_seconds} SECONDS UNTIL MIDNIGHT UTC")
        time.sleep(wait_seconds)

        try:
            log("MIDNIGHT REACHED - RELOADING MODEL")
            load_model_from_s3()
            log("MODEL RELOADED SUCCESSFULLY")
        except Exception:
            log("ERROR DURING MODEL RELOAD")
            log(traceback.format_exc())

        time.sleep(5)

# =============================
# START BACKGROUND THREAD
# =============================
def start_scheduler():
    thread = threading.Thread(target=midnight_model_reloader, daemon=True)
    thread.start()
    log("MIDNIGHT RELOADER THREAD STARTED")

# =============================
# INITIAL LOAD
# =============================
log("LOADING MODEL AT STARTUP")

load_model_from_s3()

start_scheduler()

# =============================
# REQUEST MODEL
# =============================
class PredictionRequest(BaseModel):
    prices: list[float]

# =============================
# ROOT ENDPOINT
# =============================
@app.get("/")
def root():
    return {"message": "Gold Price Prediction API running"}

# =============================
# PREDICT ENDPOINT
# =============================
@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        input_data = np.array(request.prices)

        if len(input_data) != LAGS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {LAGS} values, got {len(input_data)}"
            )

        input_data = input_data.reshape(1, -1)

        prediction = model.predict(input_data)

        return {"predicted_price": float(prediction[0])}

    except HTTPException as http_error:
        raise http_error

    except Exception:
        log("ERROR DURING PREDICTION")
        log(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

# =============================
# RUN LOCALLY
# =============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_from_s3:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )