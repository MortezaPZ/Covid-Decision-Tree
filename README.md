# 🌳 Decision Tree from Scratch — COVID-19 Dataset

Implementation and comparison of Decision Tree algorithms (CART & C4.5) trained on the COVID-19 patient dataset.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `cart.py` | CART Decision Tree (Gini Impurity) — from scratch |
| `C4.5.py` | C4.5 Decision Tree (Gain Ratio) — from scratch |
| `code_with_sklearn.py` | Baseline: scikit-learn DecisionTreeClassifier |
| `covid_trees.py` | Bonus: Circular Queue data structure |
| `Covid Data.csv` | Dataset (not included — see below) |

---

## 📊 Dataset

**COVID-19 Patient Data** — tabular medical dataset with features such as age, sex, comorbidities, and hospitalization status.

- **Target column:** `CLASIFFICATION_FINAL` (1–7 severity scale)
- **Dropped columns:** `DATE_DIED`, `ICU`, `INTUBED`, `PREGNANT`
- 📥 [Download from Kaggle — COVID-19 Dataset Mexico](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)

---

## 🚀 How to Run
```bash
pip install numpy pandas scikit-learn
```
# CART from scratch
```bash
python cart.py
```
# C4.5 from scratch
```bash
python C4.5.py
```
# Scikit-learn baseline
```bash
python code_with_sklearn.py
```
---

## 🧠 Algorithms

### CART (`cart.py`)
- Splitting criterion: **Gini Impurity**
- Binary splits on all features
- Recursive tree building with configurable `max_depth`
- Evaluated with weighted F1-score on 20% test split

### C4.5 (`C4.5.py`)
- Splitting criterion: **Gain Ratio** (Information Gain / Split Info)
- Handles both discrete and continuous features (e.g., `AGE` uses threshold)
- Multi-way splits for categorical features

### Scikit-learn Baseline (`code_with_sklearn.py`)
- Standard `DecisionTreeClassifier` for performance comparison

---

## 📈 Results

| Model | F1 Score (weighted) |
|-------|-------------------|
| CART (from scratch) | ~0.22 |
| C4.5 (from scratch) | ~0.97 |
| Scikit-learn | ~0.49 |

> Fill in actual scores after running.

---

## 👤 Author

**Morteza Pazhoum** — [@MortezaPZ](https://github.com/MortezaPZ)

K.N. Toosi University of Technology — Computer Science
