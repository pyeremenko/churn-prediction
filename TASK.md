# 🧠 Senior ML Engineer — Take-Home Assignment

Welcome! This short take-home is designed to help us understand how you approach a practical ML problem end-to-end — from data handling to being deployment-ready.

We expect this can be done **in an evening**. Please don’t over-polish — focus on **clarity, correctness, and completeness**.

---

## 🎯 Goal

Build a **simple ML model** and a **REST API** that serves predictions.

---

## 📊 Task Description

You’ll work with a small **customer churn dataset**.  
You can use any public dataset — for example:  
👉 [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Your goal is to:

1. Train a binary classifier to predict whether a customer will churn (`Yes` / `No`).
2. Expose the trained model through a simple REST API (using **FastAPI** or **Flask**).
3. Provide clear setup and usage instructions.

---

## ✅ Minimum Requirements

### 1. `train.py`

- Load the dataset (CSV).
- Perform basic preprocessing (handle NaNs, encode categorical variables, etc.).
- Train a simple model (e.g., `LogisticRegression`, `RandomForest`, `XGBoost`, etc.).
- Print accuracy, F1, or another relevant metric.
- Save the trained model to disk (`model.pkl` or similar).

### 2. `app.py`

- Use **FastAPI** or **Flask**.
- Load the saved model at startup.
- Define one endpoint:

```
POST /predict
```

**Request body example:**

```json
{
  "gender": "Female",
  "tenure": 12,
  "MonthlyCharges": 45.3,
  "Contract": "Month-to-month"
}
```

**Expected response:**

```json
{
  "churn_probability": 0.72,
  "prediction": "Yes"
}
```

### 3. `README.md`

Include:

- Environment setup (e.g. `uv/poetry/venv`)
- How to train and serve the model
- Example `curl` or Python `requests` call

---

## 🌟 Nice to Haves

If you have time, consider:

- Adding a `/health` endpoint.
- Including a basic test (`pytest`).
- Containerizing with a simple `Dockerfile`.
- Adding minimal input validation.
- Writing some thoughts about what to add and improve if you had to build this for production (e.g. into `README.md`)

---

## 🚀 Example Usage

**Train the model:**

```bash
python train.py
```

**Run the API:**

```bash
uvicorn app:app --reload
```

**Send a prediction request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"gender": "Female", "tenure": 12, "MonthlyCharges": 45.3, "Contract": "Month-to-month"}'
```

---

## 🧩 Evaluation Criteria

| Category              | Weight | What We Look For                              |
| --------------------- | ------ | --------------------------------------------- |
| **Code Quality**      | 35%    | Clean, modular, readable Python               |
| **ML Implementation** | 25%    | Sensible model choice, preprocessing, metrics |
| **API Design**        | 20%    | Working endpoint, structured responses        |
| **Documentation**     | 10%    | Clear setup and usage instructions            |
| **Initiative**        | 10%    | Thoughtfulness, small extras, effort          |

---

## 💡 Tip for Candidates

This is not about production-scale ML — we just want to see how you think, code, and structure a small, self-contained project.

Good luck and have fun 🚀