import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

df = pd.read_csv('diabetes_dataset.csv')
df.columns = [col.strip() for col in df.columns]

x = df[['Age', 'Sex', 'Body Mass Index', 'Average Blood Pressure', 'TC',
        'LDL', 'HDL', 'TCH', 'HbA1c', 'Blood Sugar Level']]
y = df['Disease Progression']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# TEST MODEL
predictions = model.predict(x_test)
accuracy_model_score = r2_score(y_test,predictions)


print(f'accuracy model : {accuracy_model_score}')




