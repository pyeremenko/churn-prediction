# Customer Churn Prediction

Binary classifier that predicts telecom customer churn, served via a REST API.


## Setup

Requires Python 3.10+. Install with uv:

```
uv sync
```


## Train

```
python train.py [path/to/dataset.csv]
```

Dataset is in `data/telco-customer-churn.csv`, but you can specify a different path as an argument.

The `train.py` script trains a Logistic Regression baseline and a Random Forest, prints a comparison, then saves the winner to `model/model.pkl` and `model/preprocessor.pkl`.


## Serve

```
uvicorn app:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `POST` | `/predict` | Predicts churn probability |


### Predict request

```json
{
  "gender": "Male",
  "tenure": 12,
  "MonthlyCharges": 45.3,
  "Contract": "Month-to-month"
}
```


### Predict response

```json
{
  "churn_probability": 0.6821,
  "prediction": "Yes"
}
```

`prediction` is `"Yes"` when `churn_probability >= 0.5`.
