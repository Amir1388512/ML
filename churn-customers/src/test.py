# Imports
from sklearn.metrics import accuracy_score

def accuracy(model, y, x) -> float:
    """
        this function should tell me about my model job

        :param model: get a model
        :param y: get target value
        :param x: get features to predict target value
        :return: accuracy score
    """

    pred = model.predict(x)
    acc = accuracy_score(y, pred)

    return acc