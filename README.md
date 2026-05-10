# Household Power Consumption Analysis using Machine Learning

## Overview

This project focuses on analyzing household electricity consumption patterns using Machine Learning techniques to help users optimize energy usage, reduce electricity costs, and promote sustainable energy habits. The system processes smart meter data, identifies inefficient consumption behavior, predicts energy usage trends, and provides actionable insights for better energy management.

The platform combines:

* Household power consumption data
* Appliance-level monitoring
* Time-series analysis
* Machine Learning prediction models
* Interactive visualization and deployment through Flask

The project was developed as part of an AI & ML initiative focused on real-world energy optimization solutions. 


# Problem Statement

Households often lack visibility into:

* Which appliances consume the most power
* Peak energy usage periods
* Wasteful energy consumption patterns
* Cost-saving opportunities

This leads to:

* High electricity bills
* Increased carbon emissions
* Inefficient energy utilization

The goal of this project is to analyze historical household energy consumption data and build a predictive system that helps users:

* Monitor energy usage
* Detect inefficiencies
* Forecast consumption patterns
* Reduce electricity costs
* Improve sustainability practices


# Objectives

* Analyze household power consumption patterns
* Detect abnormal or inefficient energy usage
* Build predictive machine learning models
* Generate actionable energy-saving recommendations
* Provide a user-friendly dashboard for visualization
* Support scalable and real-time deployment


# Dataset Information

### Dataset Source

The dataset used in this project is the **Individual Household Electric Power Consumption Dataset** from Kaggle:

[Kaggle Dataset](https://www.kaggle.com/uciml/electric-power-consumption-data-set?utm_source=chatgpt.com)

### Dataset Details

* Total Records: **2,075,259**
* Duration: **47 months**
* Data Frequency: **Minute-level measurements**
* Format: CSV
* Size: ~133 MB

### Features

| Feature               | Description                        |
| --------------------- | ---------------------------------- |
| Global Active Power   | Household active power consumption |
| Global Reactive Power | Reactive power consumption         |
| Voltage               | Voltage measurements               |
| Global Intensity      | Current intensity                  |
| Sub-metering 1        | Kitchen appliances                 |
| Sub-metering 2        | Laundry room appliances            |
| Sub-metering 3        | HVAC and water heater systems      |
| Date & Time           | Timestamp information              |


# Feature Selection

The following features were selected for model training:

| Feature             | Reason                           |
| ------------------- | -------------------------------- |
| Global Active Power | Core energy consumption metric   |
| Sub-metering 1      | Kitchen appliance usage analysis |
| Sub-metering 2      | Laundry appliance monitoring     |
| Sub-metering 3      | HVAC usage and peak analysis     |
| Datetime            | Time-series trend analysis       |

Excluded features:

* Voltage
* Global Reactive Power

Reason:
These were less useful for actionable consumer-level insights.


# Data Preprocessing

The dataset underwent extensive preprocessing including:

* Handling missing values
* Removing duplicate records
* Detecting and treating outliers
* Datetime conversion and formatting
* Feature engineering
* Data normalization and scaling

### Data Quality Issues Addressed

| Issue                     | Resolution                |
| ------------------------- | ------------------------- |
| Missing values            | Interpolation & cleaning  |
| Duplicate timestamps      | Removed duplicates        |
| Extreme outliers          | IQR-based capping         |
| Timestamp inconsistencies | Standardized formatting   |
| Sensor drift anomalies    | Rolling median correction |


# Exploratory Data Analysis (EDA)

EDA was performed to understand:

* Energy consumption trends
* Appliance-level behavior
* Peak usage patterns
* Correlations between features
* Seasonal and time-based consumption behavior

Analysis techniques included:

* Histograms
* Boxplots
* Heatmaps
* Scatter plots
* Correlation matrices
* Time-series visualization


# Machine Learning Models

Several machine learning models were evaluated for energy consumption classification and prediction.

## Final Selected Model

### Logistic Regression

The tuned Logistic Regression model was selected because it provided:

* High accuracy
* Fast prediction speed
* Better interpretability
* Scalability for deployment

### Hyperparameters

```python
{
    'penalty': 'l2',
    'C': 0.5,
    'solver': 'saga',
    'max_iter': 500,
    'class_weight': 'balanced'
}
```

# Model Performance

| Metric             | Value      |
| ------------------ | ---------- |
| Baseline Accuracy  | 82.3%      |
| Optimized Accuracy | 86.2%      |
| F1 Score           | 0.84       |
| Training Time      | 28 seconds |

### Classification Highlights

* Precision: 0.87
* Recall: 0.82
* Weighted F1 Score: 0.84


# Tech Stack

## Programming Language

* Python

## Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Matplotlib
* Seaborn
* Flask
* Jupyter Notebook

## Tools

* Git & GitHub
* Jupyter Lab


# Project Structure

```bash
Power-Consumption-Analysis/
│
├── data/
│   └── household_power_consumption.csv
│
├── notebooks/
│   └── analysis_and_model_training.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│
├── model/
│   └── dtc_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

# Web Application Deployment

The trained model was integrated with a Flask web application.

### Workflow

1. User enters consumption-related inputs
2. Flask backend processes the request
3. ML model predicts energy consumption insights
4. Results are displayed on the dashboard

Features include:

* Real-time prediction
* User-friendly interface
* Interactive analysis
* Energy-saving recommendations


# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/power-consumption-analysis.git
cd power-consumption-analysis
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```



* Real-time smart meter integration
* IoT-based appliance monitoring
* Deep learning forecasting models
* Utility pricing API integration
* Mobile application support
* Automated load shifting recommendations
* Smart home automation support


# Social & Business Impact

## Social Impact

* Helps households reduce electricity bills
* Encourages sustainable energy habits
* Promotes environmental awareness

## Business Impact

* Assists utility providers in demand forecasting
* Reduces grid strain during peak hours
* Enables data-driven energy management solutions




* Open-source Python ML community

