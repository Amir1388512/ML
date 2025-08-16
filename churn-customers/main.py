# Imports
from src import processor, analysis, model, test

df = processor.processor('./dataset/churn_train.csv')
x,y = analysis.split_data(df)

model.model.fit(x,y)

test_df = processor.processor('./dataset/churn_test.csv')
x,y = analysis.split_data(test_df)
acc = test.accuracy(model.model, y,x)
print(acc)
