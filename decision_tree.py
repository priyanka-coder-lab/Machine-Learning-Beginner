decision_tree.py

from sklearn.tree import DecisionTreeClassifier

# Data
marks = [[30], [40], [60], [70], [80], [90]]
result = [0, 0, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass

# Model
model = DecisionTreeClassifier()
model.fit(marks, result)

# Prediction
print("Prediction for 34 marks:", model.predict([[34]]))
print("Prediction for 75 marks:", model.predict([[75]]))
