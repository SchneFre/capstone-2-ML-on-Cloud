# Gold Price Predictor on Cloud (End-to-End ML System)

This project is a complete end-to-end machine learning system on AWS for predicting gold prices. It includes:

- A Streamlit Web App (Frontend)
- AWS EC2: A FastAPI Prediction Service (Backend API)
- AWS EC2: An Automated ML Pipeline (Training + Drift Detection)
- AWS S3 for data and model storage
- AWS Lambda Function to fetch gold price data

---

## System Architecture

Streamlit Frontend → sends HTTP request → FastAPI Prediction API → loads model → AWS S3 (model + data) → ML Pipeline (daily loop) → updates models/data back in S3.


---

## Components


### 1. Streamlit App (Frontend)

A simple and interactive UI for predicting gold prices.

#### Features

- Loads latest gold price data from AWS S3  
- Allows editing of last 5 days' prices  
- Sends data to FastAPI for prediction  
- Displays predicted gold price  

---

### 2. FastAPI Prediction API

A REST API running on an EC2 instance that serves predictions using a trained ML model stored in S3.

#### Features

- Loads model dynamically from S3 (`model.pkl`)
- Accepts last 5 prices
- Returns predicted next-day price
- Includes logging and error handling

---

### 3. ML Pipeline (Recurring Training System)

This is the core intelligence engine of the system. It continuously monitors data, trains models, and automatically updates them when needed.

#### Overview

The ML pipeline runs in an infinite daily loop and performs:

- Data ingestion from AWS S3  
- Model training (Linear Regression + Scaling)  
- Model evaluation (RMSE)  
- Drift detection (performance comparison)  
- Conditional retraining  
- Saving model + metrics back to S3  
- Sends email notification when retraining was neccessary
- Continuous monitoring via heartbeat logs  

---

### 4. AWS Lambda Data Ingestion (Gold Price Fetcher)

This component is responsible for automatically fetching historical gold price data from Yahoo Finance and storing it in AWS S3.
It acts as the data source for the entire ML pipeline.

#### Overview

- Fetch daily gold price data
- Convert API response into structured CSV format
- Store the dataset in AWS S3
- Provide fresh data for training and prediction systems

---
### Presentation
https://docs.google.com/presentation/d/1YVhfP0x-CZQVWcc6EUqsjH5hpGODkwS01b8IiaxIc7A/edit?usp=sharing