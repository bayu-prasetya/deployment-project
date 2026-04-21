import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

def build_bundle(pipeline: Pipeline, X_train: pd.DataFrame) -> dict:
    preprocessor           = pipeline[:-1]
    model                  = pipeline[-1]
    X_train_transformed    = preprocessor.transform(X_train)
    feature_names_raw      = list(X_train.columns)
    feature_names_transformed = preprocessor.get_feature_names_out().tolist()

    return {
        "pipeline":                  pipeline,
        "features":                  feature_names_raw,
        "feature_names_transformed": feature_names_transformed,
        "X_train_transformed":       X_train_transformed,
    }

def save_bundle(bundle: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Bundle saved → {path}")

def load_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)