import argparse
import joblib
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

console = Console(width=120)

DROP_COLS = ["customerID"]
TARGET = "Churn"

CATEGORICAL_COLS = ["gender", "Contract"]

NUMERIC_COLS = ["tenure", "MonthlyCharges"]


def render_message(title: str, text: str) -> None:
    console.rule(f"[bold cyan]{title}")
    console.print(f"  {text}")

def render_table(title: str, columns: list[tuple[str, dict]], rows: list[tuple | None]) -> None:
    """Rows may contain None as a section separator."""
    console.rule(f"[bold cyan]{title}")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    for name, opts in columns:
        t.add_column(name, **opts)
    for row in rows:
        if row is None:
            t.add_section()
        else:
            t.add_row(*[str(v) for v in row])
    console.print(t)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = df.copy()
    df = df[CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]]

    df.dropna(subset=[TARGET], inplace=True)

    numeric_medians: dict[str, float] = {}
    for col in NUMERIC_COLS:
        median = df[col].median()
        numeric_medians[col] = median
        df[col] = df[col].fillna(median)

    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = pd.Series(
        target_encoder.fit_transform(df[TARGET]),
        index=df.index,
        name=TARGET,
    )

    X = df.drop(columns=[TARGET])
    artifacts = {
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "numeric_medians": numeric_medians,
        "feature_columns": list(X.columns),
    }

    return X, y, artifacts


def show_metrics(report: dict, acc: float, f1: float, n_test: int) -> None:
    cols = [
        ("Class", {"style": "cyan"}),
        ("Precision", {"justify": "right"}),
        ("Recall", {"justify": "right"}),
        ("F1", {"justify": "right"}),
        ("Support", {"justify": "right"}),
    ]
    rows = [
        (label, f"{report[label]['precision']:.3f}", f"{report[label]['recall']:.3f}",
         f"{report[label]['f1-score']:.3f}", str(int(report[label]['support'])))
        for label in ["No Churn", "Churn"]
    ]
    rows += [
        None,
        ("[bold]Overall[/bold]", f"[bold]{acc:.3f}[/bold]", "", f"[bold]{f1:.3f}[/bold]", str(n_test)),
    ]
    render_table("Metrics", cols, rows)


def show_confusion_matrix(cm) -> None:
    render_table(
        "Confusion Matrix",
        [("", {"style": "cyan"}), ("Pred: No Churn", {"justify": "right"}), ("Pred: Churn", {"justify": "right"})],
        [
            ("Actual: No Churn", str(cm[0][0]), str(cm[0][1])),
            ("Actual: Churn",    str(cm[1][0]), str(cm[1][1])),
        ],
    )


def show_saved(model_path: Path, preprocessor_path: Path) -> None:
    render_message("Saved", f"Model         [green]{model_path}[/green]\n  Preprocessors [green]{preprocessor_path}[/green]")


def show_comparison(
    lr_acc: float,
    rf_acc: float,
    xgb_acc: float,
    lr_f1: float,
    rf_f1: float,
    xgb_f1: float,
) -> None:
    cols = [
        ("Model", {"style": "cyan"}),
        ("Accuracy", {"justify": "right"}),
        ("Macro F1", {"justify": "right"}),
    ]
    rows = [
        ("Logistic Regression (baseline)", f"{lr_acc:.3f}", f"{lr_f1:.3f}"),
        ("Random Forest", f"{rf_acc:.3f}", f"{rf_f1:.3f}"),
        ("XGBoost", f"{xgb_acc:.3f}", f"{xgb_f1:.3f}"),
    ]
    render_table("Model Comparison", cols, rows)
    winners = {
        "Logistic Regression": lr_f1,
        "Random Forest": rf_f1,
        "XGBoost": xgb_f1,
    }
    winner = max(winners, key=winners.get)
    console.print(f"\n  [bold]Winner (by macro F1):[/bold] [green]{winner}[/green]\n")


def train_baseline(X_train, X_test, y_train, y_test) -> tuple[float, float, LogisticRegression, StandardScaler]:
    render_message("Baseline", "Training Logistic Regression…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    show_metrics(report, acc, f1, len(y_test))
    return acc, f1, lr, scaler


def train_random_forest(X_train, X_test, y_train, y_test) -> tuple[float, float, RandomForestClassifier]:
    render_message("Random Forest", "Training RandomForestClassifier…")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # shape (n, 2); [:, 1] = churn_probability

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    show_metrics(report, acc, f1, len(y_test))
    show_confusion_matrix(cm)
    render_message(
        "Probability estimates",
        f"predict_proba() verified — sample churn probabilities (first 5): "
        + ", ".join(f"{p:.3f}" for p in y_proba[:5, 1]),
    )
    return acc, f1, model


def train_xgboost(X_train, X_test, y_train, y_test) -> tuple[float, float, XGBClassifier]:
    render_message("XGBoost", "Training XGBoost…")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    show_metrics(report, acc, f1, len(y_test))
    render_message(
        "Probability estimates",
        f"predict_proba() verified — sample churn probabilities (first 5): "
        + ", ".join(f"{p:.3f}" for p in y_proba[:5, 1]),
    )
    return acc, f1, model


def train(data_path: str):
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.pkl"
    preprocessor_path = model_dir / "preprocessor.pkl"

    df = load_data(data_path)
    render_message(
        "Dataset",
        f"Rows: [bold]{df.shape[0]:,}[/bold]  Columns: [bold]{df.shape[1]}[/bold]  "
        f"Churn rate: [bold]{df['Churn'].eq('Yes').mean():.1%}[/bold]",
    )
    X, y, artifacts = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_acc, lr_f1, lr_model, lr_scaler = train_baseline(X_train, X_test, y_train, y_test)
    rf_acc, rf_f1, model = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_acc, xgb_f1, xgb_model = train_xgboost(X_train, X_test, y_train, y_test)

    show_comparison(lr_acc, rf_acc, xgb_acc, lr_f1, rf_f1, xgb_f1)

    candidates = {
        "logistic_regression": {"model": lr_model, "f1": lr_f1, "scaler": lr_scaler},
        "random_forest": {"model": model, "f1": rf_f1, "scaler": None},
        "xgboost": {"model": xgb_model, "f1": xgb_f1, "scaler": None},
    }
    winner_name = max(candidates, key=lambda k: candidates[k]["f1"])
    winner = candidates[winner_name]

    artifacts["model_name"] = winner_name
    artifacts["scaler"] = winner["scaler"]

    joblib.dump(winner["model"], model_path)
    joblib.dump(artifacts, preprocessor_path)
    show_saved(model_path, preprocessor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="data/telco-customer-churn.csv", help="Path to the dataset CSV")
    args = parser.parse_args()
    train(args.data)
