import pandas as pd

df = pd.read_csv("data/telco-customer-churn.csv")

print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== head ===")
print(df.head())

print("\n=== missing values (isnull) ===")
print(df.isnull().sum())

print("\n=== blank strings per column ===")
for col in df.select_dtypes(include="object").columns:
    n = (df[col].str.strip() == "").sum()
    if n > 0:
        print(f"  {col}: {n} blank(s)")

print("\n=== Churn value_counts ===")
print(df["Churn"].value_counts())
