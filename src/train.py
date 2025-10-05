import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# Reproducibility
RANDOM_STATE = 42

def main():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = data.frame[data.feature_names]
    y: pd.Series = data.frame[data.target_names[0]] if hasattr(data, "target_names") else data.target

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Simple baseline model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}")

    # Save model and metadata
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(
        {
            "model": model,
            "feature_names": data.feature_names,
            "target_name": data.target_names[0] if hasattr(data, "target_names") else "MedHouseVal",
        },
        artifact_path,
    )

    import json
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "feature_names": data.feature_names,
                "target_name": data.target_names[0] if hasattr(data, "target_names") else "MedHouseVal",
                "rmse": rmse,
            },
            f,
            indent=2,
        )

    print(f"Saved model to {artifact_path}")

if __name__ == "__main__":
    main()