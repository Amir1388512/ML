# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

# Read Dataset
df = pd.read_csv('../datasets/tictactoe_status_dataset.csv')

# Convert My Data Into Integer
df['status'] = df['status'].map({
    'none' : 0,
    'X_win' : 1,
    'O_win' : 2,
    'draw' : -1
})

# Features Data
x = df[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']]

# Target Data
y = df['status']

# Split Data Into Test / Train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)

# Create Model
model = HistGradientBoostingClassifier()
# Train Model
model.fit(x_train, y_train)

# Model Accuracy
predict = model.predict(x_test)
accuracy = accuracy_score(y_test, predict)

# Accuracy : 99.8 %

# Save Model To Use It On Other File
dump(model, 'joblib/game_status_model.joblib')