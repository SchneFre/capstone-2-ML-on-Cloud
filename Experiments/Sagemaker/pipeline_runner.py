import subprocess
import time

print("STEP 1: Running ingestion script")

subprocess.run(["python3", "yahoo-to-s3.py"], check=True)

print("INGESTION COMPLETE")

print("Waiting 30 seconds...")
time.sleep(30)

print("STEP 2: Running training script")

subprocess.run(["python3", "training-to-s3.py"], check=True)

print("PIPELINE COMPLETE")