import numpy as np
import pandas as pd
import boto3
import io
import os
import pickle
import json
import time
from datetime import datetime

from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# LOGGER (PRINT VERSION)
# =============================
def log(msg):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}", flush=True)

log("🚀 STARTING S3-ONLY ML PIPELINE (DAILY LOOP MODE)")

# =============================
# ENV VARIABLES
# =============================
load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
PREFIX = os.getenv("PREFIX", "gold-prices/")

MODEL_KEY = os.getenv("MODEL_KEY", "models/model.pkl")
RMSE_KEY = os.getenv("RMSE_KEY", "models/rmse.json")

LAGS = int(os.getenv("LAGS", 5))
THRESHOLD_PERCENT = float(os.getenv("THRESHOLD_PERCENT", 0.10))

# =============================
# AWS CLIENT
# =============================
log("☁️ Connecting to S3")
s3_client = boto3.client("s3")

# =============================
# LOAD LATEST DATA
# =============================
def load_latest_data():
    log("📥 Loading latest dataset from S3")

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = response.get("Contents", [])

    if not files:
        raise Exception("❌ No data found in S3")

    latest_file = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
    file_key = latest_file["Key"]

    log(f"📄 Using file: {file_key}")

    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
    csv_data = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(io.StringIO(csv_data))

    if "close" not in df.columns:
        raise Exception("❌ close column missing")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    data = df["close"].values

    X, y = [], []

    for i in range(LAGS, len(data)):
        X.append(data[i - LAGS:i])
        y.append(data[i])

    return np.array(X), np.array(y)

# =============================
# BUILD MODEL
# =============================
def build_model():
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LinearRegression())
    ])

# =============================
# SAVE MODEL TO S3
# =============================
def save_model(model, rmse):
    log("💾 Saving model to S3 (overwrite)")

    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=MODEL_KEY,
        Body=buffer.getvalue()
    )

    rmse_data = {
        "rmse": rmse,
        "timestamp": str(datetime.utcnow())
    }

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=RMSE_KEY,
        Body=json.dumps(rmse_data).encode("utf-8")
    )

    log(f"✅ Model saved: {MODEL_KEY}")
    log(f"📉 RMSE saved: {rmse}")

# =============================
# LOAD MODEL FROM S3
# =============================
def load_model():
    try:
        log("📦 Checking if model exists in S3")

        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        model_data = obj["Body"].read()

        model = pickle.loads(model_data)

        log("✅ Existing model loaded from S3")
        return model

    except Exception:
        log("⚠️ No existing model found in S3")
        return None

# =============================
# LOAD PREVIOUS RMSE
# =============================
def load_previous_rmse():
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=RMSE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data["rmse"]

    except Exception:
        log("⚠️ No previous RMSE found")
        return None

# =============================
# TRAIN MODEL
# =============================
def train_model(X, y):
    log("🏋️ Training model")

    model = build_model()
    model.fit(X, y)

    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)

    log(f"📊 TRAIN METRICS | RMSE={rmse:.4f}, R2={r2:.4f}")

    return model, rmse

# =============================
# MAIN LOOP (DAILY PIPELINE)
# =============================
log("🔁 ENTERING CONTINUOUS DAILY LOOP")

while True:
    try:
        log("🟢 NEW PIPELINE CYCLE STARTED")

        X, y = load_latest_data()

        model = load_model()
        previous_rmse = load_previous_rmse()

        # =============================
        # CASE 1: NO MODEL EXISTS
        # =============================
        if model is None:
            log("🆕 No model exists → training new model")

            model, rmse = train_model(X, y)
            save_model(model, rmse)

        # =============================
        # CASE 2: MODEL EXISTS
        # =============================
        else:
            log("🔍 Evaluating existing model")

            preds = model.predict(X)

            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)

            log(f"📊 CURRENT RMSE={rmse:.4f}, PREVIOUS RMSE={previous_rmse}")

            # =============================
            # DRIFT CHECK
            # =============================
            if previous_rmse is None:
                log("⚠️ No baseline RMSE → updating baseline")
                save_model(model, rmse)

            else:
                degraded = rmse > previous_rmse * (1 + THRESHOLD_PERCENT)

                if degraded:
                    log("⚠️ MODEL PERFORMANCE DECREASED → retraining")

                    model, new_rmse = train_model(X, y)
                    save_model(model, new_rmse)

                else:
                    log("✅ Model still good → no retraining needed")

        log("🏁 PIPELINE CYCLE FINISHED")

    except Exception as e:
        log(f"❌ ERROR IN PIPELINE: {str(e)}")

    # =============================
    # DAILY SLEEP + HEARTBEAT
    # =============================
    sleep_seconds = 86400  # 24 hours
    heartbeat_interval = 300  # 5 minutes

    log("⏳ Sleeping for 24 hours (heartbeat every 5 minutes)")

    elapsed = 0

    while elapsed < sleep_seconds:
        log("💓 HEARTBEAT: pipeline still running")
        time.sleep(heartbeat_interval)
        elapsed += heartbeat_interval