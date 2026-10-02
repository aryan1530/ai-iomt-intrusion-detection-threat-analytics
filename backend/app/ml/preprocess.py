from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DROP_ALWAYS = {"difficulty", "dataset"}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    for c in list(df.columns):
        if df[c].isna().mean() > 0.95:
            df.drop(columns=[c], inplace=True)
    for c in DROP_ALWAYS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = clean_dataframe(df)
    y_bin = df["binary_label"].astype(str)
    y_cat = df["attack_category"].astype(str)
    X = df.drop(columns=[c for c in ["binary_label", "attack_category", "label", "class", "Label"] if c in df.columns])
    return X, y_bin, y_cat


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in {"category", "bool"}]
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def fit_transform_split(X: pd.DataFrame, y_bin: pd.Series, y_cat: pd.Series, test_size: float = 0.2, seed: int = 42):
    X_train, X_test, yb_train, yb_test, yc_train, yc_test = train_test_split(
        X, y_bin, y_cat, test_size=test_size, random_state=seed, stratify=y_bin
    )
    pre = build_preprocessor(X_train)
    Xt_train = pre.fit_transform(X_train)
    Xt_test = pre.transform(X_test)
    names = list(pre.get_feature_names_out())
    bin_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    yb_tr = bin_enc.fit_transform(yb_train)
    yb_te = bin_enc.transform(yb_test)
    yc_tr = cat_enc.fit_transform(yc_train)
    yc_te = cat_enc.transform(yc_test)
    return {
        "preprocessor": pre,
        "feature_names": names,
        "X_train": Xt_train,
        "X_test": Xt_test,
        "yb_train": yb_tr,
        "yb_test": yb_te,
        "yc_train": yc_tr,
        "yc_test": yc_te,
        "bin_enc": bin_enc,
        "cat_enc": cat_enc,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


def save_preproc(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": bundle["preprocessor"],
            "feature_names": bundle["feature_names"],
            "bin_enc": bundle["bin_enc"],
            "cat_enc": bundle["cat_enc"],
        },
        path,
    )
    (path.parent / "feature_names.json").write_text(json.dumps(bundle["feature_names"], indent=2))
