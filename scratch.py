import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=120)

def render_message(title: str, text: str) -> None:
    console.rule(f"[bold cyan]{title}")
    console.print(f"  {text}")


def render_table(title: str, columns: list[tuple[str, dict]], rows: list[tuple]) -> None:
    console.rule(f"[bold cyan]{title}")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    for name, opts in columns:
        t.add_column(name, **opts)
    for row in rows:
        t.add_row(*[str(v) for v in row])
    console.print(t)


def show_shape(df: pd.DataFrame) -> None:
    render_message("Shape", f"Rows: [bold]{df.shape[0]:,}[/bold]  Columns: [bold]{df.shape[1]}[/bold]")


def show_dtypes(df: pd.DataFrame) -> None:
    render_table(
        "Column Types",
        [("Column", {"style": "cyan"}), ("Dtype", {})],
        list(df.dtypes.items()),
    )


def show_missing(df: pd.DataFrame) -> None:
    s = df.isnull().sum()
    missing = s[s > 0]
    if missing.empty:
        render_message("Missing Values (isnull)", "[green]No NaN values found.[/green]")
    else:
        render_table(
            "Missing Values (isnull)",
            [("Column", {"style": "cyan"}), ("NaN Count", {"justify": "right"})],
            list(missing.items()),
        )


def show_blanks(df: pd.DataFrame) -> None:
    blanks = {
        col: int((df[col].str.strip() == "").sum())
        for col in df.select_dtypes(include="object").columns
    }
    blanks = {k: v for k, v in blanks.items() if v > 0}
    if not blanks:
        render_message("Blank Strings", "[green]No blank strings found.[/green]")
    else:
        render_table(
            "Blank Strings",
            [("Column", {"style": "cyan"}), ("Blank Count", {"justify": "right", "style": "yellow"})],
            list(blanks.items()),
        )


def show_churn(df: pd.DataFrame) -> None:
    counts = df["Churn"].value_counts()
    pcts = df["Churn"].value_counts(normalize=True).mul(100).round(1)
    render_table(
        "Churn Distribution",
        [("Churn", {"style": "cyan"}), ("Count", {"justify": "right"}), ("Percent", {"justify": "right"})],
        [(label, f"{counts[label]:,}", f"{pcts[label]}%") for label in counts.index],
    )


df = pd.read_csv("data/telco-customer-churn.csv")

show_shape(df)
show_dtypes(df)
show_missing(df)
show_blanks(df)
show_churn(df)
