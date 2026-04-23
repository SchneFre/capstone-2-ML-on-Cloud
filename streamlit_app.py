import streamlit as st
import boto3
import pandas as pd
import io
import requests
import os
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-gold-price-fjs")
PREFIX = os.getenv("PREFIX", "gold-prices/")

API_URL = "http://54.198.181.155:8000/predict"

# =========================
# S3 CLIENT
# =========================
s3 = boto3.client("s3")

# =========================
# LOAD LAST FILE FROM S3
# =========================
def get_last_5_days():
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = response.get("Contents", [])

    if not files:
        return [0, 0, 0, 0, 0]

    latest_file = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
    key = latest_file["Key"]

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    df = pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8")))

    if "close" not in df.columns:
        raise ValueError("Missing 'close' column in dataset")

    closes = df["close"].dropna().values

    return closes[-5:].tolist()


# =========================
# STREAMLIT UI
# =========================
st.title("💰 Gold Price Predictor (Next Day)")

st.write("Edit the last 5 days of gold prices and predict today's price.")

# Load data
try:
    last_5 = get_last_5_days()
except Exception as e:
    st.error(f"Error loading S3 data: {e}")
    last_5 = [0, 0, 0, 0, 0]

st.subheader("📊 Last 5 Days Prices")

prices = []

for i in range(5):
    val = st.number_input(
        f"Day {i+1}",
        value=float(last_5[i]) if i < len(last_5) else 0.0,
        step=0.1
    )
    prices.append(val)

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Today's Price"):

    payload = {"prices": prices}

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()["predicted_price"]

            st.success(f"🎯 Predicted Gold Price: {result:.2f}")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Request failed: {e}")