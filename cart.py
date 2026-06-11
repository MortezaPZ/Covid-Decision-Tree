import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def gini_impurity_node(y):
    _, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob ** 2)


def gini_gain(y, y_left, y_right):
    n = len(y)
    weighted = (len(y_left) / n) * gini_impurity_node(y_left) + \
               (len(y_right) / n) * gini_impurity_node(y_right)
    return gini_impurity_node(y) - weighted


def split(X, y, feature_index, threshold):
    left = X[:, feature_index] <= threshold
    return X[left], y[left], X[~left], y[~left]


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.leaf = True


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # FIX: was n_classes + 1, causing index out of range for last class
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=gini_impurity_node(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        if (self.max_depth is None or depth < self.max_depth) and len(set(y)) > 1:
            best_gain, best_index, best_threshold = 0, None, None
            for feature_index in range(X.shape[1]):
                for threshold in np.unique(X[:, feature_index]):
                    X_l, y_l, X_r, y_r = split(X, y, feature_index, threshold)
                    if len(y_l) > 0 and len(y_r) > 0:
                        gain = gini_gain(y, y_l, y_r)
                        if gain > best_gain:
                            best_gain, best_threshold, best_index = gain, threshold, feature_index
            if best_gain > 0:
                node.feature_index = best_index
                node.threshold = best_threshold
                X_l, y_l, X_r, y_r = split(X, y, best_index, best_threshold)
                node.leaf = False
                node.left = self._build_tree(X_l, y_l, depth + 1)
                node.right = self._build_tree(X_r, y_r, depth + 1)
        return node

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self.tree
        while not node.leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node.predicted_class


# Load and preprocess
data = pd.read_csv('Covid Data.csv')
data['PREGNANT'] = data['PREGNANT'].replace([97, 98, 99], 2)
data = data.drop(columns=['DATE_DIED', 'ICU', 'INTUBED'])

X = data.drop(columns='CLASIFFICATION_FINAL').values
y = data['CLASIFFICATION_FINAL'].values

# FIX: proper train/test split for honest evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)  # limit depth for speed
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
