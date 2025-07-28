from model import *
from sklearn.metrics import r2_score

predictions = model.predict(x_test)
acc = r2_score(y_test, predictions)

if __name__ == '__main__':
    print(f'model accuracy : {acc}')