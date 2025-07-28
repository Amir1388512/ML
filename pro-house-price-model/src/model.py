from preprocess import x, y
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

x_train , x_test,  y_train , y_test = train_test_split(x ,y , test_size=0.33, random_state=42)
model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1, max_depth=4)
model.fit(x_train, y_train)

dump(model, '../model/house-price-model.joblib')