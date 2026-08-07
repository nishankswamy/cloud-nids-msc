# Cloud-Native ML Network Intrusion Detection — Local Development

COM7014 Advanced Computing Project — Nishank Swamy

## Project Structure

```
nids_project/
├── data/
│   ├── raw/            # Place downloaded CICIDS2017 CSVs here
│   └── processed/      # Cleaned/preprocessed output lands here
├── src/
│   ├── load_data.py        # Load + combine CICIDS2017 CSV files
│   ├── explore_data.py     # Exploratory data analysis (class balance, feature overview)
│   ├── preprocess.py       # Cleaning, encoding, feature selection, train/test split
│   ├── train_models.py     # Train + compare Random Forest, XGBoost, Neural Net
│   └── evaluate.py         # Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
├── models/              # Saved trained model files (.pkl)
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
```

## Getting the Dataset

1. Download CICIDS2017 (recommended: cleaned version) from Kaggle:
   https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
2. Place the CSV file(s) into `data/raw/`
3. Run the pipeline in order (see below)

## Running the Pipeline

```bash
python src/load_data.py        # Step 1: combine/load raw CSVs
python src/explore_data.py     # Step 2: EDA — class distribution, feature stats
python src/preprocess.py       # Step 3: clean, encode, balance, split
python src/train_models.py     # Step 4: train Random Forest, XGBoost, Neural Net
python src/evaluate.py         # Step 5: compare models, print metrics, save best model
```

## Notes

- All scripts use `RAW_DATA_DIR` and `PROCESSED_DATA_DIR` constants at the top — update these if your folder layout differs.
- `train_models.py` saves each trained model to `models/` as a `.pkl` file via `joblib`.
- These scripts were validated against a small synthetic sample matching the CICIDS2017 schema (see `src/generate_synthetic_sample.py`) to confirm the pipeline logic runs end-to-end before you run it against the real ~2.8M-row dataset.
