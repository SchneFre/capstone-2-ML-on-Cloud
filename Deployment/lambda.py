import json
import boto3
import urllib.request
import csv
from datetime import datetime

s3 = boto3.client('s3')

BUCKET_NAME = "s3-gold-price-fjs"
S3_KEY = "gold-prices/gold_prices.csv"

def lambda_handler(event, context):
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1y&interval=1d"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

        result = data["chart"]["result"][0]

        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        opens = quote["open"]
        highs = quote["high"]
        lows = quote["low"]
        closes = quote["close"]
        volumes = quote["volume"]

        csv_data = [["date", "open", "high", "low", "close", "volume"]]

        for i in range(len(timestamps)):
            if closes[i] is None:
                continue

            date = datetime.utcfromtimestamp(timestamps[i]).strftime('%Y-%m-%d')

            csv_data.append([
                date,
                opens[i],
                highs[i],
                lows[i],
                closes[i],
                volumes[i]
            ])

        file_path = "/tmp/gold_prices.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        # Updated S3 path + filename
        s3.upload_file(file_path, BUCKET_NAME, S3_KEY)

        return {
            "statusCode": 200,
            "body": json.dumps(f"Saved {len(csv_data)-1} rows to {S3_KEY}")
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(str(e))
        }