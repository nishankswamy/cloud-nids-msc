"""
Step 2: Exploratory Data Analysis (EDA).
Prints class distribution, missing values, and basic feature statistics.
Use findings here to justify preprocessing decisions in the report's Fact Finding section.
"""
import pandas as pd
import os

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

LABEL_COL = "Label"


def explore(df: pd.DataFrame):
    print("=" * 60)
    print("SHAPE:", df.shape)

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION (raw counts and %)")
    counts = df[LABEL_COL].value_counts()
    pct = df[LABEL_COL].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "pct": pct.round(2)})
    print(summary)

    print("\n" + "=" * 60)
    print("MISSING VALUES (top 10 columns)")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10))

    print("\n" + "=" * 60)
    print("INFINITE VALUES (numeric columns)")
    numeric_df = df.select_dtypes(include="number")
    inf_counts = (numeric_df == float("inf")).sum() + (numeric_df == float("-inf")).sum()
    print(inf_counts[inf_counts > 0])

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS:", df.duplicated().sum())

    print("\n" + "=" * 60)
    print("FEATURE COUNT (excluding label):", df.shape[1] - 1)

    print("\n" + "=" * 60)
    imbalance_ratio = counts.max() / counts.min()
    print(f"CLASS IMBALANCE RATIO (largest:smallest class): {imbalance_ratio:.1f} : 1")
    if imbalance_ratio > 10:
        print(">> Significant imbalance detected. Consider SMOTE, class weighting, or undersampling in preprocess.py")

    return summary


if __name__ == "__main__":
    path = os.path.join(PROCESSED_DATA_DIR, "combined_raw.parquet")
    df = pd.read_parquet(path)
    explore(df)
