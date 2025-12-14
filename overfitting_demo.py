overfitting.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

marks = [[30], [40], [50], [60], [70], [80], [90]]
result = [0, 0, 0, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    marks, result, test_size=0.3
)

model = LogisticRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
