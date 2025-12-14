ml_intro.py

from sklearn.linear_model import LinearRegression

# Data
marks = [[30], [60], [80], [90]]
result = [0, 1, 1, 1]   # 0 = Fail, 1 = Pass

# Model
model = LinearRegression()
model.fit(marks, result)

# Prediction
print("Prediction for 70 marks:", model.predict([[70]]))
