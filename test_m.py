import pickle
import numpy as np

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

print("Total symptoms:", len(columns))

# Create input (all 0 first)
input_data = [0] * len(columns)

# Manually set some symptoms
# Example: fever, headache
for i, col in enumerate(columns):
    if col in ["fever", "headache"]:
        input_data[i] = 1

# Predict
prediction = model.predict([input_data])

print("Predicted Disease:", prediction[0])