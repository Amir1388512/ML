# Imports
import pandas as pd
from sklearn.impute import SimpleImputer


def processor(file_name) -> pd.DataFrame:
    """
        this func should clean and sort my datasets

        :param file_name: the url of dataset
        :return: a clean pandas dataframe
    """
    df = pd.read_csv(file_name)

    # we fill the empty col with most frequent strategy
    smp_imp = SimpleImputer(strategy='most_frequent')
    df[['internet_service']] = smp_imp.fit_transform(df[['internet_service']])

    # select columns which has object type
    cols = [col for col in df.columns if df[col].dtype == object]
    for col in cols :
        # we replace object with some coded number
        df[col] = df[col].factorize()[0]

    return df
