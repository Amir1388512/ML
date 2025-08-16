# Imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump


# Read File
df = pd.read_csv('../dataset/housePrice.csv')

# Create sample from simple imputer to fill the missed value
smp_imp = SimpleImputer(strategy='constant', fill_value='Unknown')
df[['Address']] = smp_imp.fit_transform(df[['Address']])

# find columns with datatype object and boolean
not_numeric_cols = [col for col in df.columns if df[col].dtype == object or df[col].dtype == bool ]
# Remove Area From This List
# Area columns has some value with , and some are int
not_numeric_cols.remove('Area')
df['Area'] = df['Area'].str.replace(',', '')
df['Area'] = df['Area'].astype('int64')

# convert categorical value into integer
for col in not_numeric_cols:
    df[col] = df[col].factorize()[0]

# Create Some Columns
# This Columns Can Help My Model To Predict Better Than Then
df['PricePerMeter'] = df['Area'] / df['Price']
df['PricePerRoom']  = df['Room'] / df['Price']

x = df[['Area','Room', 'Parking', 'Warehouse', 'Elevator', 'Address', 'PricePerMeter', 'PricePerRoom']]

# scale my data
smp_scaler = StandardScaler()
x = smp_scaler.fit_transform(x)
y = df['Price']

# split my data into train and test
x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# create my model
model = HistGradientBoostingRegressor()
model.fit(x_train, y_train)

# test / score
predict = model.predict(x_test)
# my Model Accuracy is around 98%
accuracy = r2_score(y_test, predict)

# Save Model
dump(model, 'tehran_house_price.joblib')






