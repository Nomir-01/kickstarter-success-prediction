# Kickstarter Campaign Success Prediction

A machine-learning analysis of Kickstarter campaign outcomes using only information available at campaign launch. The canonical pipeline compares Logistic Regression, Decision Tree, Random Forest, optional XGBoost, and an optional Bayesian Network.

Outcome variables such as pledged amount and backer count are deliberately excluded to prevent target leakage.

## Repository structure

```text
.
├── src/train_models.py        # Canonical reproducible pipeline
├── data/                      # Local dataset location (CSV is Git-ignored)
├── notebooks/                 # Original exploratory Colab notebooks
├── reports/                   # Written report
├── presentation/              # Presentation source and export
└── references/                # Original dataset and Colab links
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The dataset is included at `data/kickstarter_projects.csv`. Its original source link is also recorded in `references/Colab & Dataset Links.pdf`.

## Run

Run the three core models:

```bash
python src/train_models.py
```

For a quicker development check:

```bash
python src/train_models.py --sample 20000
```

Include the two more computationally expensive models:

```bash
python src/train_models.py --include-xgboost --include-bayesian
```

Results are printed and written to `artifacts/model_results.csv`.

## Reproducibility corrections

The canonical pipeline corrects the issues present in the original notebook state:

- Every estimator and metric is explicitly imported.
- Categorical encoding is fitted inside each model pipeline.
- Every model uses the same stratified split.
- Bayesian discretization, structure learning, and parameter fitting use training data only.
- Colab-specific uploads are not required.
- Logistic Regression has a sufficient iteration limit and XGBoost uses current parameters.

## Documents

The PDF presentation contains four more slides than the PowerPoint source. Both original files are retained so no coursework artifact is lost; use the PDF when reviewing the complete submitted presentation.
