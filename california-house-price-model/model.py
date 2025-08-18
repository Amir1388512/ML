# Imports
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


# Create My Model
class HousePriceRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, features, targets):
        x = np.array(features)
        y = np.array(targets).reshape(-1, 1)

        x_b = np.hstack([x, np.ones((x.shape[0], 1))])
        theta = np.linalg.pinv(x_b).dot(y).flatten()

        self.w = theta[:-1]
        self.b = theta[-1]
        return self

    def predict(self, features):
        x = np.array(features)
        return x.dot(self.w) + self.b


# Read Dataset
df = pd.read_csv("housing.csv")

# Fill Missing Values
smp_imp = SimpleImputer(strategy="median")
df[["total_bedrooms"]] = smp_imp.fit_transform(df[["total_bedrooms"]])

# Convert Categorical Value Into Integer
df["ocean_proximity"] = df["ocean_proximity"].factorize()[0]

# Create New Features (No Leakage)
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["population_per_household"] = df["population"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

# Features
x = df[
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room",
    ]
]

# Target
y = df["median_house_value"]

# Scale Features
smp_scaler = StandardScaler()
x_scaled = smp_scaler.fit_transform(x)

# Split Data Into Train/Test
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.33, random_state=42
)

# Train Model
model = HousePriceRegressor()
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)

# Evaluate
accuracy = r2_score(y_test, predictions)
print("R² Score:", accuracy)
