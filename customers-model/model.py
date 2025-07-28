import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df["Contract"] = df["Contract"].map({"Month-to-month": 0, "Two year": 1, "One year": 2})


columns_with_yes = [
    col for col in df.columns if df[col].astype(str).str.contains("Yes").any()
]
for index in columns_with_yes:
    df[index] = df[index].map({"Yes": 1, "No": 0})
null_col = [col for col in df.columns if df[col].isnull().any()]
sample_imputer = SimpleImputer(strategy="most_frequent")
for index in null_col:
    df[index] = sample_imputer.fit_transform(df[[index]])
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

x = df[
    [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
    ]
]

y = df["Churn"]


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
acc = classification_report(y_test, predictions)

print(acc)
