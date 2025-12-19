import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from scipy import sparse


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


def to_dense_if_needed(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


def run_preprocessing(input_csv: str, output_dir: str, target_col: str,
                      test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(input_csv)
    assert target_col in df.columns, f"target_col '{target_col}' tidak ada di kolom dataset."

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_p = preprocessor.transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    X_train_d = to_dense_if_needed(X_train_p)
    X_test_d  = to_dense_if_needed(X_test_p)

    out_cat_cols = []
    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        out_cat_cols = ohe.get_feature_names_out(cat_cols).tolist()

    out_cols = num_cols + out_cat_cols

    train_df = pd.DataFrame(X_train_d, columns=out_cols)
    test_df  = pd.DataFrame(X_test_d, columns=out_cols)

    train_df[target_col] = y_train.values
    test_df[target_col]  = y_test.values

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path  = os.path.join(output_dir, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train_path, test_path = run_preprocessing(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("OK. Saved:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()
