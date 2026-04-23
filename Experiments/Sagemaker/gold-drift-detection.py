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
# CONFIGURATION
# =============================
EXPERIMENT_NAME = "Gold Price Prediction"
LAGS = 5

BUCKET_NAME = "s3-gold-price-fjs"
PREFIX = "gold-prices/"

THRESHOLD_PERCENT = 0.10  # 10 percent degradation threshold

mlflow.set_experiment(EXPERIMENT_NAME)

# =============================
# GET LATEST MODEL FROM MLFLOW
# =============================
def get_latest_run_id():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

    if runs.empty:
        raise Exception("No MLflow runs found")

    runs = runs[runs["status"] == "FINISHED"]

    if "start_time" not in runs.columns:
        raise Exception("start_time column not found in MLflow runs")

    runs = runs.dropna(subset=["start_time"])

    # Sort by most recent run
    runs = runs.sort_values("start_time", ascending=False)

    latest_run = runs.iloc[0]

    run_id = latest_run["run_id"]

    if "metrics.rmse" in runs.columns and not pd.isna(latest_run.get("metrics.rmse", None)):
        rmse = latest_run["metrics.rmse"]
    else:
        rmse = None

    print("Latest training run id:", run_id)
    print("Baseline RMSE from latest run:", rmse)

    return run_id, rmse


run_id, baseline_rmse = get_latest_run_id()

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Model loaded successfully")

# =============================
# LOAD LATEST DATA FROM S3
# =============================
s3_client = boto3.client("s3")

response = s3_client.list_objects_v2(
    Bucket=BUCKET_NAME,
    Prefix=PREFIX
)

files = response.get("Contents", [])

if not files:
    raise Exception("No files found in S3 bucket")

latest_file = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
file_key = latest_file["Key"]

print("Drift evaluation file:", file_key)

obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
csv_data = obj["Body"].read().decode("utf-8")

df = pd.read_csv(io.StringIO(csv_data))

# =============================
# CLEAN DATA
# =============================
print("Raw columns:", df.columns)

if "Close" not in df.columns:
    raise Exception("Close column not found in dataset")

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

data = df["Close"].astype(float).values

# =============================
# CREATE LAG FEATURES
# =============================
X, y = [], []

for i in range(LAGS, len(data)):
    X.append(data[i - LAGS:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise Exception("Not enough data for evaluation")

# =============================
# PREDICT WITH EXISTING MODEL
# =============================
predictions = model.predict(X)

mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print("\n=============================")
print("DRIFT EVALUATION RESULTS")
print("=============================")
print("New RMSE:", rmse)
print("R2 score:", r2)
print("Baseline RMSE:", baseline_rmse)

# =============================
# DRIFT DETECTION
# =============================
degradation_detected = False

if baseline_rmse is not None:
    degradation_detected = rmse > baseline_rmse * (1 + THRESHOLD_PERCENT)

if degradation_detected:
    print("\nPERFORMANCE DEGRADATION DETECTED")
    print("Triggering automatic retraining...")

    # =============================
    # RETRAIN MODEL AUTOMATICALLY
    # =============================
    print("\nSTARTING RETRAINING PIPELINE")

    test_size = 30

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    retrain_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LinearRegression())
    ])

    with mlflow.start_run(run_name="automatic_retraining_after_drift"):

        retrain_pipeline.fit(X_train, y_train)

        new_predictions = retrain_pipeline.predict(X_test)

        new_mse = mean_squared_error(y_test, new_predictions)
        new_rmse = np.sqrt(new_mse)
        new_r2 = r2_score(y_test, new_predictions)

        mlflow.log_param("lags", LAGS)
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("trigger", "drift_retraining")

        mlflow.log_metric("mse", new_mse)
        mlflow.log_metric("rmse", new_rmse)
        mlflow.log_metric("r2", new_r2)

        mlflow.sklearn.log_model(
            retrain_pipeline,
            artifact_path="model",
            input_example=X_train[:1]
        )

        print("\nRETRAINING COMPLETE")
        print("New RMSE:", new_rmse)
        print("New R2:", new_r2)

else:
    print("\nNO SIGNIFICANT DEGRADATION")
    print("Model remains stable.")