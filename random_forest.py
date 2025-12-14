random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
marks = [[30], [40], [60], [70], [80], [90]]
result = [0, 0, 1, 1, 1, 1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    marks, result, test_size=0.3
)

# Model
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
