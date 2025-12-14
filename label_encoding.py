label_encoding.py

from sklearn.preprocessing import LabelEncoder

cities = ["Pune", "Mumbai", "Pune", "Nashik"]

encoder = LabelEncoder()
encoded_city = encoder.fit_transform(cities)

print("Original:", cities)
print("Encoded:", encoded_city)
