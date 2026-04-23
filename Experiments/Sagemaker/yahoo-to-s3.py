import yfinance as yf
import boto3
import pandas as pd
import io
from datetime import datetime, timedelta

# ================================
# 1. DEFINE DATE RANGE (FULL YEAR)
# ================================
gold = yf.Ticker("GC=F")
gold_data = gold.history(period="1y")

print(gold_data.tail())

# ================================
# 4. S3 CLIENT
# ================================
s3_client = boto3.client("s3")

bucket_name = "s3-gold-price-fjs"

# ================================
# 5. FIXED FILE NAME (OVERWRITE SAME FILE)
# ================================
s3_key = "gold-prices/gold_prices.csv"

# ================================
# 6. CONVERT TO CSV IN MEMORY
# ================================
csv_buffer = io.StringIO()
gold_data.to_csv(csv_buffer, index=False)

# ================================
# 7. UPLOAD TO S3
# ================================
s3_client.put_object(
    Bucket=bucket_name,
    Key=s3_key,
    Body=csv_buffer.getvalue()
)

print("Upload successful (FULL YEAR DATA)")
print(f"s3://{bucket_name}/{s3_key}")