import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=120)
df = pd.read_csv("data/telco-customer-churn.csv")


console.rule("[bold cyan]Shape")
console.print(f"  Rows: [bold]{df.shape[0]:,}[/bold]  Columns: [bold]{df.shape[1]}[/bold]")


console.rule("[bold cyan]Column Types")
t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
t.add_column("Column", style="cyan")
t.add_column("Dtype")
for col, dtype in df.dtypes.items():
    t.add_row(col, str(dtype))
console.print(t)


console.rule("[bold cyan]Missing Values (isnull)")
missing = df.isnull().sum()
if missing.sum() == 0:
    console.print("  [green]No NaN values found.[/green]")
else:
    mt = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    mt.add_column("Column", style="cyan")
    mt.add_column("NaN Count", justify="right")
    for col, n in missing[missing > 0].items():
        mt.add_row(col, str(n))
    console.print(mt)


console.rule("[bold cyan]Blank Strings")
blanks = {
    col: int((df[col].str.strip() == "").sum())
    for col in df.select_dtypes(include="object").columns
}
blanks = {k: v for k, v in blanks.items() if v > 0}
if not blanks:
    console.print("  [green]No blank strings found.[/green]")
else:
    bt = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    bt.add_column("Column", style="cyan")
    bt.add_column("Blank Count", justify="right", style="yellow")
    for col, n in blanks.items():
        bt.add_row(col, str(n))
    console.print(bt)


console.rule("[bold cyan]Churn Distribution")
vc = df["Churn"].value_counts()
pct = df["Churn"].value_counts(normalize=True).mul(100).round(1)
ct = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
ct.add_column("Churn", style="cyan")
ct.add_column("Count", justify="right")
ct.add_column("Percent", justify="right")
for label in vc.index:
    ct.add_row(label, f"{vc[label]:,}", f"{pct[label]}%")
console.print(ct)
