from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow
import mlflow.sklearn

EXPERIMENT_NAME = "Gold Price Prediction"

app = FastAPI()

# =============================
# LOAD LATEST MODEL
# =============================
def get_latest_run_id():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

    if runs.empty:
        raise Exception("No MLflow runs found")

    runs = runs[runs["status"] == "FINISHED"]
    runs = runs.sort_values("start_time", ascending=False)

    return runs.iloc[0]["run_id"]

run_id = get_latest_run_id()

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Loaded run:", run_id)
print("Model type:", type(model))
print("EXPECTED INPUT SHAPE:", model.named_steps["scaler"].n_features_in_)

# =============================
# REQUEST MODEL
# =============================
class PredictionRequest(BaseModel):
    prices: list[float]

# =============================
# ROUTES
# =============================
@app.get("/")
def root():
    return {"message": "Gold Price Prediction API running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    print("entering predict:")
    # -------------------------
    # STEP 1: FORCE CLEAN ARRAY
    # -------------------------
    input_array = np.asarray(request.prices)

    print("RAW INPUT:", request.prices)
    print("NP ARRAY SHAPE BEFORE FIX:", input_array.shape)

    # -------------------------
    # STEP 2: VALIDATE STRUCTURE
    # -------------------------
    input_array = input_array.flatten()

    if input_array.shape[0] != 5:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 5 values, got {input_array.shape[0]}"
        )

    # -------------------------
    # STEP 3: FINAL MODEL INPUT
    # -------------------------
    input_array = input_array.reshape(1, 5)

    print("FINAL SHAPE:", input_array.shape)

    # -------------------------
    # STEP 4: PREDICT
    # -------------------------
    prediction = model.predict(input_array)

    return {
        "predicted_price": float(prediction[0])
    }

# =============================
# RUN
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)