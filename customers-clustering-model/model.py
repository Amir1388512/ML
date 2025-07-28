import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from joblib import dump


df = pd.read_csv('Mall_Customers.csv')


x = df[['Spending Score (1-100)', 'Annual Income (k$)']]

sse = []

for i in range(1,20):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(x)
    sse.append(model.inertia_)
k_values = range(1, 20)
best_k_value = KneeLocator(k_values, sse, curve="convex", direction="decreasing")


pipe = Pipeline([
    ('scaler' , StandardScaler()),
    ('model', KMeans(n_clusters=best_k_value.elbow, random_state=42))
])

pipe.fit(x)
df['Cluster'] = pipe.fit_predict(x)
acc = silhouette_score(x,df['Cluster'])

dump(pipe, 'Customers-Clustering.joblib')
df.to_csv('new_data.csv')
