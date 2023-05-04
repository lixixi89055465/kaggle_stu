from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(["paris", "paris", "tokyo", "amsterdam"])
print(encoder.transform(["paris", "tokyo", "amsterdam"]))
print(encoder.inverse_transform([1, 1, 0, 2, 2, 2]))
print("0"*100)
print(encoder.classes_)