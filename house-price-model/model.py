import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



df = pd.read_csv('house_prices_dataset.csv')

x = df[['area','rooms', 'floor', 'age', 'elevator', 'location_score']]
y = df[['price']]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

prediction_test = model.predict(x_test)
res = r2_score(y_test, prediction_test)
print(f'accuracy = {res * 100} %')



