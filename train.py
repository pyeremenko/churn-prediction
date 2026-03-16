import argparse
import joblib
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

console = Console(width=120)

DROP_COLS = ["customerID"]
TARGET = "Churn"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


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
    df.drop(columns=DROP_COLS, inplace=True)

    # TotalCharges is stored as string; whitespace entries (tenure == 0) become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

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


def show_saved(model_path: Path, preprocessors_path: Path) -> None:
    render_message("Saved", f"Model         [green]{model_path}[/green]\n  Preprocessors [green]{preprocessors_path}[/green]")


def train(data_path: str):
    p = Path(data_path)
    model_path = p.with_name(p.stem + "-model.pkl")
    preprocessors_path = p.with_name(p.stem + "-preprocessors.pkl")

    df = load_data(data_path)
    X, y, artifacts = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    show_metrics(report, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"), len(y_test))
    show_confusion_matrix(cm)

    joblib.dump(model, model_path)
    joblib.dump(artifacts, preprocessors_path)
    show_saved(model_path, preprocessors_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default=None, help="Path to the dataset CSV")
    args = parser.parse_args()
    train(args.data)
