import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# Data Visualizing
# ---------------------------------------------------------------------------------------------
df = pd.read_csv('datasets/train.csv')
age_imp = SimpleImputer(strategy='mean')
df['Age'] = age_imp.fit_transform(df[['Age']])
Embarked_imp = SimpleImputer(strategy='most_frequent')
df['Embarked'] = Embarked_imp.fit_transform(df[['Embarked']]).ravel()
sample_ohe = OneHotEncoder()
gender_encoded = sample_ohe.fit_transform(df[['Sex']]).toarray()
gender_df = pd.DataFrame(
            gender_encoded,
            index=df.index,
            columns=sample_ohe.get_feature_names_out(['Sex'])
)
Embarked_encoded = sample_ohe.fit_transform(df[['Embarked']]).toarray()
Embarked_df = pd.DataFrame(
            Embarked_encoded,
            index=df.index,
            columns=sample_ohe.get_feature_names_out(['Embarked'])
)
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)

df = pd.concat([df, Embarked_df, gender_df], axis=1)
# ---------------------------------------------------------------------------------------------

# Test/Train Data
# ---------------------------------------------------------------------------------------------
X = df[['Pclass','Has_Cabin','Sex_male','Sex_female','Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df['Survived']
x_train , x_test, y_train,y_test = train_test_split(X_scaled,y, test_size=0.33, random_state=42)
# ---------------------------------------------------------------------------------------------

# Model Creation
# ---------------------------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000,random_state=42, class_weight='balanced')
model.fit(x_train, y_train)
# ---------------------------------------------------------------------------------------------

# Model Test
# ---------------------------------------------------------------------------------------------
pred_y = model.predict(x_test)
model_accuracy = classification_report(y_test ,pred_y)   # 81%
print(model_accuracy)




# ---------------------------------------------------------------------------------------------