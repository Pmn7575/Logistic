import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

df = pd.read_csv('Diabetis.csv', header=None, names=columns)

X = df[['Glucose']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

x = np.sort(X['Glucose'].values)
y_curve = model.predict_proba(x.reshape(-1, 1))[:, 1]

plt.scatter(X['Glucose'], y)
plt.plot(x, y_curve)
plt.xlabel("Glucose")
plt.ylabel("Probability")
plt.title("Logistic Regression Sigmoid Curve")
plt.show()