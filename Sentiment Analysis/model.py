# Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('sentiment_dataset.csv')
df['sentiment'] = df['sentiment'].factorize()[0]
smp_vector = TfidfVectorizer()
txt = df['sentence']
X_tfidf = smp_vector.fit_transform(df['sentence'])
X = pd.DataFrame(X_tfidf.toarray(), columns=smp_vector.get_feature_names_out())
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
predict = model.predict(x_test)

acc = classification_report(y_test, predict)
# model overfitted Bcuz my dataset is very short and not support all options in sentences
