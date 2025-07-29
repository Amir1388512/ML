import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv('data.csv')
sample_ohe = OneHotEncoder(sparse_output=False)
categorical_car = sample_ohe.fit_transform(df[['Car_Type']])
car_df  = pd.DataFrame(categorical_car, columns=sample_ohe.get_feature_names_out(['Car_Type']))
df = pd.concat([df.select_dtypes(include='number'), car_df], axis=1)

x = df.drop(columns=['Price_million_IRR'])
s = StandardScaler()
x = s.fit_transform(x)

y = df['Price_million_IRR']

x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)

acc = r2_score(y_test, pred)
print(acc)