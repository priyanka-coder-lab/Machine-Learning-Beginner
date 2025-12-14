feature_target.py

from sklearn.linear_model import LogisticRegression

# Features: Marks, Study Hours
X = [
    [30, 1],
    [40, 2],
    [60, 4],
    [70, 5],
    [80, 6],
    [90, 7]
]

# Target: Pass(1) / Fail(0)
y = [0, 0, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print("Prediction (50 marks, 3 hrs):", model.predict([[50, 3]]))
print("Prediction (85 marks, 6 hrs):", model.predict([[85, 6]]))
