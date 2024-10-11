# Hybrid Models with Logistic Regression, Random Forest, and SHAP for assessing recidivism risk.

This repository contains an implementation of hybrid machine learning models for analyzing the COMPAS dataset, focusing on the recidivism prediction task. The project demonstrates how to leverage both Logistic Regression for interpretability and Random Forest for complex pattern detection, and it provides insights into model performance using SHAP for explainability.

## Dataset

The COMPAS dataset is a widely used dataset for assessing recidivism risk, which includes various demographic and criminal history features. The relevant features used in this project are:

- `age`: Age of the individual.
- `race`: Race of the individual.
- `sex`: Gender of the individual.
- `juv_fel_count`: Number of juvenile felony counts.
- `juv_misd_count`: Number of juvenile misdemeanor counts.
- `priors_count`: Number of prior counts.
- `c_charge_degree`: Charge degree.

The target variable is `two_year_recid`, which indicates whether an individual was re-arrested within two years.

## Installation

To run this project, clone the repository and install the required dependencies. You can use the following command:

```bash
pip install -r requirements.txt
```

## Usage

Download the COMPAS dataset and place the CSV file in the project directory.
Update the path to the CSV file in the script.
Run the hybrid_models_Compas_Shap.py script to train the models and evaluate their performance.
```bash
python hybrid_models_Compas_Shap.py
```

## Results
The script evaluates the performance of the Logistic Regression model, Random Forest model, and a hybrid model combining both. It also visualizes the performance metrics and provides SHAP visualizations for interpretability.
