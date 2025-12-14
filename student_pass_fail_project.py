student_pass_fail_project.py

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Dataset
marks = [80, 40, 90, 60, 30]
hours = [5, 2, 6, 4, 1]
cities = ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai"]
result = [1, 0, 1, 1, 0]   # 1 = Pass, 0 = Fail

# Encode city
encoder = LabelEncoder()
city_encoded = encoder.fit_transform(cities)

# Features & Target
X = list(zip(marks, hours, city_encoded))
y = result

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[70, 4, encoder.transform(["Pune"])[0]]])
print("Prediction for student:", prediction)

# Accuracy
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
