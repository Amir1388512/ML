import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data.csv")
df["diagnosis"]  = df["diagnosis"].map({"M": 0, "B": 1})

important_data = df.corr()['diagnosis'].abs().drop('diagnosis').loc[lambda x: x > 0.1].index.tolist()
f = 3
x = df[important_data]
sample_scaler = StandardScaler()
x = sample_scaler.fit_transform(x)
y = df['diagnosis']

x_train ,x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

pred = model.predict(x_test)
acc = classification_report(y_test, pred)
print(acc)


