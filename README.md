# Customer Churn Prediction

## Setup

Requires Python 3.10+. Install with uv:

```
uv sync
```

## Train

```
python train.py [path/to/dataset.csv]
```

Dataset path is optional — defaults to `data/telco-customer-churn.csv`. Saves `{BASE}-model.pkl` and `{BASE}-preprocessors.pkl` alongside the CSV.
