"""Reproducible Kickstarter campaign-success modelling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
NUMERIC_FEATURES = ["Goal_log", "Duration", "Launch_month", "Launch_year"]
CATEGORICAL_FEATURES = ["Category", "Subcategory", "Country"]


def load_data(path: Path, sample: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "Category", "Subcategory", "Country", "Launched", "Deadline",
        "Goal", "State",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    state = df["State"].astype(str).str.strip().str.lower()
    df = df[state.isin(["successful", "failed"])].copy()
    df["Target"] = (state.loc[df.index] == "successful").astype(int)
    df["Launched"] = pd.to_datetime(df["Launched"], errors="coerce")
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    df["Duration"] = (df["Deadline"] - df["Launched"]).dt.days
    df["Goal_log"] = np.log1p(pd.to_numeric(df["Goal"], errors="coerce"))
    df["Launch_month"] = df["Launched"].dt.month
    df["Launch_year"] = df["Launched"].dt.year
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Target"])
    df = df[df["Duration"] > 0].reset_index(drop=True)

    if sample and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def evaluate(name: str, model: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probability),
    }


def evaluate_bayesian(train: pd.DataFrame, test: pd.DataFrame, limit: int = 1000) -> dict:
    """Fit structure, bins and parameters using training data only."""
    from pgmpy.estimators import BayesianEstimator, BicScore, HillClimbSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianNetwork

    discretizer = KBinsDiscretizer(
        n_bins=[10, 4], encode="ordinal", strategy="quantile", subsample=None
    )
    train_bins = discretizer.fit_transform(train[["Goal_log", "Duration"]]).astype(int)
    test_bins = discretizer.transform(test[["Goal_log", "Duration"]]).astype(int)

    top_subcategories = set(train["Subcategory"].value_counts().head(30).index)
    top_countries = set(train["Country"].value_counts().head(20).index)

    def bn_frame(source: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "Category": source["Category"].astype(str).to_numpy(),
            "Subcategory": source["Subcategory"].where(
                source["Subcategory"].isin(top_subcategories), "Other"
            ).astype(str).to_numpy(),
            "Country": source["Country"].where(
                source["Country"].isin(top_countries), "Other"
            ).astype(str).to_numpy(),
            "Goal_bin": bins[:, 0].astype(str),
            "Duration_bin": bins[:, 1].astype(str),
            "Target": source["Target"].astype(int).to_numpy(),
        })

    train_bn = bn_frame(train, train_bins)
    test_bn = bn_frame(test, test_bins).head(limit)
    structure_sample = train_bn.sample(
        min(50_000, len(train_bn)), random_state=RANDOM_STATE
    )
    structure = HillClimbSearch(structure_sample).estimate(
        scoring_method=BicScore(structure_sample), show_progress=False
    )
    model = BayesianNetwork(structure.edges())
    model.add_nodes_from(train_bn.columns)
    model.fit(
        train_bn,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )
    inference = VariableElimination(model)
    fallback = float(train_bn["Target"].mean())
    probabilities = []
    for _, row in test_bn.iterrows():
        evidence = row.drop("Target").to_dict()
        try:
            query = inference.query(["Target"], evidence=evidence, show_progress=False)
            state_index = query.state_names["Target"].index(1)
            probabilities.append(float(query.values[state_index]))
        except (KeyError, ValueError):
            probabilities.append(fallback)

    y_true = test_bn["Target"].to_numpy()
    probability = np.asarray(probabilities)
    prediction = (probability >= 0.5).astype(int)
    return {
        "Model": "BayesianNetwork",
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/kickstarter_projects.csv"))
    parser.add_argument("--sample", type=int, help="Optional stratified-development sample size")
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--include-bayesian", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data, args.sample)
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["Target"],
    )
    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train["Target"]
    y_test = test["Target"]

    estimators = {
        "LogisticRegression": LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    if args.include_xgboost:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    results = []
    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ])
        results.append(evaluate(name, pipeline, x_train, x_test, y_train, y_test))

    if args.include_bayesian:
        results.append(evaluate_bayesian(train, test))

    result_frame = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    result_frame.to_csv(output_dir / "model_results.csv", index=False)
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
