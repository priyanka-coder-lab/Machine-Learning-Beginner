model_evaluation.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Data
marks = [[30], [40], [60], [70], [80], [90]]
result = [0, 0, 1, 1, 1, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    marks, result, test_size=0.3
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
