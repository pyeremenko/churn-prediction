import joblib
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

console = Console(width=120)

DATA_PATH = "data/telco-customer-churn.csv"
MODEL_PATH = "data/model.pkl"
PREPROCESSORS_PATH = "data/preprocessors.pkl"

DROP_COLS = ["customerID"]
TARGET = "Churn"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
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


def train():
    df = load_data()
    X, y, artifacts = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)


    console.rule("[bold cyan]Metrics")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    t.add_column("Class", style="cyan")
    t.add_column("Precision", justify="right")
    t.add_column("Recall", justify="right")
    t.add_column("F1", justify="right")
    t.add_column("Support", justify="right")
    for label in ["No Churn", "Churn"]:
        r = report[label]
        t.add_row(label, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", str(int(r['support'])))
    t.add_section()
    t.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{accuracy_score(y_test, y_pred):.3f}[/bold]",
        "",
        f"[bold]{f1_score(y_test, y_pred, average='macro'):.3f}[/bold]",
        str(len(y_test)),
    )
    console.print(t)


    console.rule("[bold cyan]Confusion Matrix")
    cm_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    cm_table.add_column("", style="cyan")
    cm_table.add_column("Pred: No Churn", justify="right")
    cm_table.add_column("Pred: Churn", justify="right")
    cm_table.add_row("Actual: No Churn", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("Actual: Churn", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)


    joblib.dump(model, MODEL_PATH)
    joblib.dump(artifacts, PREPROCESSORS_PATH)
    console.rule("[bold cyan]Saved")
    console.print(f"  Model        [green]{MODEL_PATH}[/green]")
    console.print(f"  Preprocessors [green]{PREPROCESSORS_PATH}[/green]")


if __name__ == "__main__":
    train()
