import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove unwanted columns if any (like duplicate or wrong names)
data = data.loc[:, ~data.columns.duplicated()]

# Features (all except diseases)
X = data.drop("diseases", axis=1)

# Target
y = data["diseases"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save column names (VERY IMPORTANT)
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("Model trained with full dataset successfully!")