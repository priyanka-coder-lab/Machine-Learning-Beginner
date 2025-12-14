ml_train_test.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data
marks = [[30], [40], [60], [70], [80], [90]]
result = [0, 0, 1, 1, 1, 1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    marks, result, test_size=0.2
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:", y_test)
