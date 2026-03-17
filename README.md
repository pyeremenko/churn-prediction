# Customer Churn Prediction

Binary classifier that predicts telecom customer churn, served via a REST API.

Training:

<img width="1491" height="1385" alt="image" src="https://github.com/user-attachments/assets/474cd23b-3d92-43e4-b8f9-697f9664584a" />


Running:

<img width="1987" height="714" alt="image" src="https://github.com/user-attachments/assets/2703c857-3aac-499b-9d82-1f5534cecbc4" />

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

__

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


## TODO

The current implementation is a proof of concept. For production, the following improvements should be made:

### Input validation

- The current schema accepts any float for `MonthlyCharges` and any string for `Contract`. Add Pydantic validators with allowed enum values, numeric range checks, and a reject-on-unknown-fields policy to prevent silent GIGO.

### Model versioning

- Tag each saved artifact with a version (e.g. `model_v1.2.pkl`) and store the git SHA + training timestamp inside the pickle as metadata

### Prediction logging

- Persist every request/response pair to a database (Postgres + SQLAlchemy works well): input features, predicted probability, prediction label, model version, and a UUID for later feedback tying. This can help tp detect drift and allow to audit the predictions.

### Monitoring

- *Latency & errors*: expose a `/metrics` endpoint (Prometheus + Grafana) tracking p50/p95/p99 inference latency and 4xx/5xx rates.

### CI/CD

- Add a GitHub Actions pipeline: 
  - lint 
  - unit tests
  - build & push Docker image tagged with the commit hash
  - deploy to staging

### A/B testing 

- Route a configurable percentage of traffic (via a `X-Model-Version` haeder) to a challenger model. Log the modle version on every prediction record so outcomes can be compared once enough samples accumulate.

### Authentication & rate limiting

- Add an API key authentication and per-key rate limiting.

### ML Improvements

- Try to replace LabelEncoder with one‑hot encoding
- Bundle all preprocessing (cleaning + encoding) into one thing. Save it, so training and prediction use identical logic and move preprocessing into a single pipeline object saved to disk.
- Add better evaluation metrics, not just accuracy. Measure how well the model seprates classes (ROC-AUC) and how good it is on imbalanced data (PR-AUC)

