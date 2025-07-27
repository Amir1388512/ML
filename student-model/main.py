import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report

data_frame = pd.read_csv('datasets/student_exam_data.csv')

x = data_frame[['Study Hours', 'Previous Exam Score']]
y = data_frame['Pass/Fail']

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(model.predict([[5, 99]]))
