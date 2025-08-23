# Imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# Read Dataset
df = pd.read_csv("ai4i2020.csv")

# Create Sample Of OneHotEncoder
smp_ohe = OneHotEncoder()
# Encode Data
data_encoder = smp_ohe.fit_transform(df[["Type"]]).toarray()

df_encoder = pd.DataFrame(data_encoder, columns=smp_ohe.get_feature_names_out())
df = pd.concat([df, df_encoder], axis=1)


# Features Data
x = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_H",
        "Type_L",
        "Type_M",
    ]
]

# Create Sample Of Scaler
smp_scaler = StandardScaler()
x = smp_scaler.fit_transform(x)

# Target Data
y = df["Machine failure"]

# Split Data Into Test / Train
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=42)

# Create Model
model = HistGradientBoostingClassifier(class_weight='balanced')
# Train Model
model.fit(x_train, y_train)

# Test Model
predictions = model.predict(x_test)

# Model Accuracy
print(classification_report(y_test, predictions))

