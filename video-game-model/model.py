import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


df = pd.read_csv("data.csv")
strategies = ["most_frequent", "mean"]
columns_imp = ["Genre", "Year"]
for s, c in zip(strategies, columns_imp):
    sample = SimpleImputer(strategy=s)
    df[[c]] = sample.fit_transform(df[[c]])
obj_to_int_col = ["Genre", "Publisher", "Platform"]
for i in obj_to_int_col:
    df[i] = df[i].factorize()[0]

remove_cols = ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
x = df[
    [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in remove_cols
    ]
]
y = df["Global_Sales"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
model = GradientBoostingRegressor()
model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = r2_score(y_test, pred)  # accuracy = 0.8907954269812192
