# Customer Churn Prediction

Binary classifier that predicts telecom customer churn, served via a REST API.


## Quickstart

It's possible to test the app with Docker:

```bash
docker run -p 8000:8000 pyeremenko/churn:latest
```

Then do one of 2 things:

- open `http://localhost:8000/docs` in your browser to see the API documentation.
- use curl or any HTTP client to make requests to the API:

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"gender": "Female", "tenure": 12, "MonthlyCharges": 45.3, "Contract": "Month-to-month"}'
```

The trained model is built into the Docker image, so you don't need to train it yourself.¹

________
¹ But that's for demonstration purposes - in production you'd want to train it yourself and then build your own Docker image with the model mounted from a volume.


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
