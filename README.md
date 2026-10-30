# Decision Trees on the COVID-19 Mexico Dataset

A learning project comparing custom CART and C4.5 decision trees with a scikit-learn baseline on the public COVID-19 Mexico dataset.

## Dataset

The CSV is not included. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) and place the file at the repository root as `Covid Data.csv`.

The target column used by the scripts is `CLASIFFICATION_FINAL`. The implementations preprocess the source columns differently; review each script before comparing results. Do not commit downloaded records or other health data.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bash
python -m pip install numpy pandas scikit-learn
```

## Run

Run each script from the repository root:

```bash
python cart.py
python C4.5.py
python code_with_sklearn.py
```

The scripts print a weighted F1 score for their own evaluation setup. Results depend on the downloaded dataset version and preprocessing; this repository does not publish a reproducible benchmark table.

## Implementations

- `cart.py` builds a binary decision tree using Gini impurity.
- `C4.5.py` implements a gain-ratio tree with threshold splits for numeric features.
- `code_with_sklearn.py` provides the scikit-learn baseline.
- `covid_trees.py` contains a separate circular-queue exercise.

This is an educational exercise, not a clinical model or a source of medical advice.
