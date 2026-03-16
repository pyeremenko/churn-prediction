import pandas as pd

df = pd.read_csv("data/telco-customer-churn.csv")

print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== head ===")
print(df.head())
