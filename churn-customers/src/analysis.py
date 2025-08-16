# Imports
from sklearn.preprocessing import StandardScaler


def pick_items(df) -> list:
    """
        this function should tell me which item in dataset has good correlation for my model
        :param df: my datasets
        :return: best item in dataset
    """
    best_items = []
    corr = df.corr()['churn']
    for i in zip(corr, df.columns):
        if 0.015 < i[0] or i[0]<  -0.01:
            if i[1] != 'churn':
                best_items.append(i[1])

    return best_items

def split_data(df):
    """
        this function should to split my data into target (y) and features (x) data

        :param df: get dataframe
        :return: x , y data target data and features
    """
    smp_scaler = StandardScaler()
    best_items = pick_items(df)
    x = df[best_items]
    x = smp_scaler.fit_transform(x)
    y = df['churn']

    return x, y