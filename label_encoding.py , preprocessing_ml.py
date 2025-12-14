label_encoding.py

from sklearn.preprocessing import LabelEncoder

cities = ["Pune", "Mumbai", "Pune", "Nashik"]

encoder = LabelEncoder()
encoded_city = encoder.fit_transform(cities)

print("Original:", cities)
print("Encoded:", encoded_city)

preprocessing_ml.py


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Data
cities = ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai"]
marks = [80, 45, 90, 70, 60]
result = [1, 0, 1, 1, 1]  # Pass/Fail

# Encode city
encoder = LabelEncoder()
city_encoded = encoder.fit_transform(cities)

# Features
X = list(zip(city_encoded, marks))
y = result

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction
test_city = encoder.transform(["Pune"])
print("Prediction:", model.predict([[test_city[0], 75]]))
