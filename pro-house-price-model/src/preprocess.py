import pandas as pd
from sklearn.impute import SimpleImputer



df = pd.read_csv('../datasets/train.csv')

null_cols = df.columns[df.isnull().any()]

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

for col in null_cols:
    if df[col].dtype == object:
        df[col] = cat_imputer.fit_transform(df[[col]]).ravel()
    else:
        df[col] = num_imputer.fit_transform(df[[col]]).ravel()


obj_cols = df.select_dtypes(include='object').columns

for col in obj_cols:
    df[col] = df[col].astype('category').cat.codes

correlation = dict(df.corr()['SalePrice'])
accepted_correlation = {}
for i in correlation:
    if correlation[i] > 0.2:
        accepted_correlation = accepted_correlation | {i: correlation[i]}

accepted_correlation.pop('SalePrice')

x = df[list(accepted_correlation.keys())]

y = df['SalePrice']



