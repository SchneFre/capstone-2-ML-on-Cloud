import numpy as np
import pandas as pd
import boto3
import io
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# CONFIG
# =============================
EXPERIMENT_NAME = "Gold Price Prediction"
LAGS = 5

mlflow.set_experiment(EXPERIMENT_NAME)

# =============================
# 1. LOAD DATA FROM S3
# =============================
bucket_name = "s3-gold-price-fjs"
prefix = "gold-prices/"

s3_client = boto3.client("s3")

response = s3_client.list_objects_v2(
    Bucket=bucket_name,
    Prefix=prefix
)

files = response.get("Contents", [])

if not files:
    raise Exception("No files found in S3 bucket")

latest_file = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
file_key = latest_file["Key"]

print("Loading file:", file_key)

obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
csv_data = obj["Body"].read().decode("utf-8")

df = pd.read_csv(io.StringIO(csv_data))

# =============================
# 2. CLEAN DATA 
# =============================

print("Raw columns:", df.columns)

# Ensure correct column exists
if "Close" not in df.columns:
    raise Exception(f"'Close' column not found. Columns are: {df.columns}")

# Force numeric conversion 
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Drop bad rows (NaN from conversion issues)
df = df.dropna(subset=["Close"])

# Extract clean series
data = df["Close"].astype(float).values

print("Clean data sample:", data[:5])

# =============================
# 3. CREATE LAG FEATURES
# =============================
X, y = [], []

for i in range(LAGS, len(data)):
    X.append(data[i - LAGS:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

# =============================
# 4. TRAIN/TEST SPLIT
# =============================
test_size = 30

if len(X) <= test_size:
    raise Exception("Not enough data for training/testing split")

X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# =============================
# 5. MODEL PIPELINE
# =============================
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", LinearRegression())
])

# =============================
# 6. MLflow TRAINING
# =============================
with mlflow.start_run(run_name="linear_regression_s3_cleaned"):

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("lags", LAGS)
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("source", "S3")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        input_example=X_train[:1]
    )

    print("Training complete using CLEAN S3 data")