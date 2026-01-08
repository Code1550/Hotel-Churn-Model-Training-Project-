# Hotel Booking Cancellation Prediction

An end-to-end machine learning pipeline for predicting hotel booking cancellations using Databricks, PySpark, and MLflow.

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Monitoring](#monitoring)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project implements a production-ready ML pipeline for predicting hotel booking cancellations. Built on Databricks Serverless with Unity Catalog, the solution follows a medallion architecture (Bronze → Silver → Gold) and leverages MLflow for complete experiment tracking and model management.

**Key Features:**
- Medallion architecture with Delta Lake
- Unity Catalog for data governance
- MLflow integration for MLOps
- Comprehensive model evaluation and business impact analysis
- Built-in monitoring and retraining strategy

## Business Problem

Hotel booking cancellations create significant challenges for the hospitality industry:

- **Revenue Loss**: Last-minute cancellations reduce occupancy and revenue
- **Operational Inefficiency**: Difficulty in inventory management
- **Resource Waste**: Missed opportunities for rebooking

**Solution**: Predict high-risk cancellations to enable:
- Proactive retention strategies
- Optimized overbooking policies
- Reduced revenue loss

## Dataset

**Source**: [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle)

**Statistics**:
- 119,390 booking records
- 32 features
- Data from 2015-2017
- Two hotel types: Resort and City

**Key Features**:
| Feature | Description |
|---------|-------------|
| `lead_time` | Days between booking and arrival |
| `adr` | Average daily rate |
| `deposit_type` | Type of deposit made |
| `previous_cancellations` | Customer's cancellation history |
| `total_of_special_requests` | Number of special requests |

## Architecture

### Technology Stack

```
Databricks Serverless    →  Unified analytics platform
Unity Catalog           →  Data governance
PySpark                 →  Distributed processing
MLflow                  →  MLOps lifecycle
Delta Lake              →  Data lakehouse
```

### Medallion Architecture

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Bronze  │  →   │ Silver  │  →   │  Gold   │
│  (Raw)  │      │(Cleaned)│      │(Features)│
└─────────┘      └─────────┘      └─────────┘
```

### Notebook Details

#### 1️ Data Ingestion
**File**: `01_data_ingestion.ipynb`

- Loads CSV from `/FileStore/Tables/hotel_bookings.csv`
- Defines schema for type safety
- Validates data quality
- **Output**: `hotel_catalog.bronze.raw_hotel_bookings`

#### 2️ Data Cleaning
**File**: `02_data_cleaning.ipynb`

- Handles missing values (imputation & filling)
- Removes duplicates and outliers
- Creates derived features (`total_nights`, `total_guests`)
- **Output**: `hotel_catalog.silver.cleaned_hotel_bookings`

#### 3️ Feature Engineering
**File**: `03_feature_engineering.ipynb`

Creates 18 predictive features:
- **Numerical**: Scaled lead_time, adr, total_nights
- **Categorical**: Hotel, deposit type, customer type encodings
- **Binary**: Weekend stay, room match, special requests
- **Interactions**: Lead time × ADR, nights × guests

**Output**: `hotel_catalog.gold.hotel_features_final`

> **Note**: Manual transformations used to avoid Serverless memory constraints

#### 4️ Model Training
**File**: `04_model_training.ipynb`

**Models Trained**:
- Logistic Regression (with class weighting)
- Random Forest (lightweight)
- Gradient Boosted Trees

**Features**:
- Train/validation/test split (70/15/15)
- MLflow experiment tracking
- Class imbalance handling with stratified sampling
- Model registration in MLflow

#### 5️ Model Evaluation
**File**: `05_model_evaluation.ipynb`

**Comprehensive Analysis**:
- Statistical metrics (AUC-ROC, AUC-PR, F1-Score)
- Business metrics (ROI, saved revenue)
- Fairness analysis across segments
- Threshold optimization
- Monitoring setup with Delta tables

## Setup & Installation

### Prerequisites

- Databricks workspace with Serverless compute
- Unity Catalog enabled
- MLflow access

### Initial Setup

**1. Create Unity Catalog Structure**
```sql
CREATE CATALOG IF NOT EXISTS hotel_catalog;
USE CATALOG hotel_catalog;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
```

**2. Upload Dataset**
- Download dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Upload `hotel_bookings.csv` to `/FileStore/Tables/` in Databricks
- Verify file permissions

**3. Configure MLflow**
```python
import mlflow
mlflow.set_experiment("/Shared/hotel_churn_prediction")
```

## Usage

### Running the Pipeline

Execute notebooks in order:

```bash
# 1. Ingest raw data
01_data_ingestion.ipynb

# 2. Clean and preprocess
02_data_cleaning.ipynb

# 3. Engineer features
03_feature_engineering.ipynb

# 4. Train models
04_model_training.ipynb

# 5. Evaluate and deploy
05_model_evaluation.ipynb
```

### Making Predictions

```python
import mlflow

# Load registered model
model = mlflow.pyfunc.load_model("models:/hotel_churn_predictor/Production")

# Make predictions
predictions = model.predict(features_df)
```

## Results

### Expected Performance

| Metric | Target Range | Status |
|--------|--------------|--------|
| AUC-ROC | 0.70-0.85 |  Achievable |
| AUC-PR | 0.60-0.75 | Good for imbalanced data |
| Precision | 0.65-0.80 | Business optimized |

### Business Impact

**Assumptions**:
- Average booking value: $150
- High-risk bookings: Top 20% by predicted probability
- Prevention rate: 10% of high-risk cancellations

**Expected ROI**:
- Significant revenue recovery even with conservative prevention rates
- Reduced operational costs from better inventory management
- Improved customer satisfaction through proactive engagement

## Monitoring

### Monitoring Strategy

The project includes built-in monitoring with Delta tables:

```sql
-- Daily prediction accuracy
SELECT 
    prediction_date,
    accuracy,
    auc_roc
FROM hotel_catalog.gold.model_monitoring_predictions
WHERE prediction_date >= current_date() - 30;
```

**Key Metrics Tracked**:
- Daily prediction volumes
- Model accuracy trends
- Data drift indicators
- Business value metrics

### Retraining Strategy

| Trigger | Action |
|---------|--------|
| **Monthly** | Scheduled retraining with new data |
| **Performance drop** | Immediate retraining if AUC drops >5% |
| **Data drift** | Investigate and retrain if confirmed |

## Future Improvements

### Short-term
- [ ] Add seasonal and temporal features
- [ ] Implement XGBoost with proper Serverless configuration
- [ ] Business-driven threshold optimization

### Medium-term
- [ ] Real-time prediction streaming pipeline
- [ ] A/B testing framework for interventions
- [ ] Automated CI/CD for model updates

### Long-term
- [ ] Multi-hotel chain deployment
- [ ] Integration with weather and event data
- [ ] Causal inference for understanding cancellation drivers

## Troubleshooting

### Common Issues

**Problem**: Model predicts all zeros
```
Cause: Class imbalance without weighting
Fix: Enable class weights in Notebook 04
```

**Problem**: Memory errors in feature engineering
```
Cause: Too many features or complex transformations
Fix: Simplify features, use manual transformations instead of Spark ML
```

**Problem**: MLflow experiment errors
```
Cause: Serverless JVM access limitations
Fix: Use simple experiment names, avoid JVM-dependent operations
```

**Problem**: Model registration failures
```
Cause: Permissions or model size issues
Fix: Check MLflow permissions, simplify model architecture
```

## Project Artifacts

### Delta Tables Created
```
hotel_catalog.bronze.raw_hotel_bookings
hotel_catalog.silver.cleaned_hotel_bookings
hotel_catalog.gold.hotel_features_final
hotel_catalog.gold.model_evaluation_summary
hotel_catalog.gold.model_monitoring_predictions
```

### MLflow Artifacts
```
Experiment: /Shared/hotel_churn_prediction
Model: hotel_churn_predictor
Versions: Multiple with performance metrics
```

## Key Learnings

### Challenges Overcome

1. **Serverless Constraints**: Adapted to JVM access limitations by using manual transformations
2. **Class Imbalance**: Implemented class weighting and stratified sampling
3. **Memory Limitations**: Simplified feature engineering to respect 100MB model size limit

### Best Practices Implemented

- Data governance with Unity Catalog
- Experiment tracking with MLflow
- Modular notebook architecture
- Built-in monitoring and alerting
- Comprehensive documentation

## License

This project is for educational and experience purposes. The dataset is publicly available from Kaggle under the CC0: Public Domain license.

## References

- [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)


---

**Built with using Databricks, PySpark, and MLflow**
