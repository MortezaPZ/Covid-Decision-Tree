import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Covid Data.csv')
data['PREGNANT'] = data['PREGNANT'].replace([97, 98, 99], 2)
data = data.drop(columns=['DATE_DIED', 'ICU', 'INTUBED'])

X = data.drop(columns='CLASIFFICATION_FINAL').values
y = data['CLASIFFICATION_FINAL'].values

# FIX: use train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Scikit-Learn F1 Score:", f1_score(y_test, y_pred, average='weighted'))
