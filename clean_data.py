import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

# Create a folder to store the data
Path("data").mkdir(exist_ok=True)

# Download the public Student Performance dataset
dataset = fetch_ucirepo(id=320)

# Combine features and target columns
df = pd.concat(
    [dataset.data.features, dataset.data.targets],
    axis=1
)

# Save the original dataset
df.to_csv("data/student_performance_raw.csv", index=False)

print("Original dataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum().sum())

# Standardize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "_")
)

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
duplicates_removed = before - len(df)

# Clean text columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    df[col] = df[col].str.lower()

# Convert selected columns to numeric
numeric_columns = [
    "age", "absences", "G1", "G2", "G3"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values when present
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

# Save cleaned dataset
df.to_csv("data/student_performance_cleaned.csv", index=False)

# Create a simple cleaning report
with open("cleaning_report.txt", "w", encoding="utf-8") as report:
    report.write("SWYNEX Internship - Task 1\n")
    report.write("Data Cleaning and Preparation Report\n\n")
    report.write(f"Rows after cleaning: {len(df)}\n")
    report.write(f"Columns: {len(df.columns)}\n")
    report.write(f"Duplicate rows removed: {duplicates_removed}\n")
    report.write(
        f"Missing values remaining: {df.isnull().sum().sum()}\n"
    )

print("\nCleaning completed!")
print("Duplicate rows removed:", duplicates_removed)
print("Missing values remaining:", df.isnull().sum().sum())
print("Cleaned dataset saved in the data folder.")
