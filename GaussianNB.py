import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence']

relevant_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))
df[relevant_features] = scaler.fit_transform(df[relevant_features].fillna(0))

unique_genres = df['track_genre'].unique()
genre_random_values = {genre: np.random.rand() for genre in unique_genres} 
df['genre_random'] = df['track_genre'].map(genre_random_values)

df['liked'] = ((df[relevant_features].sum(axis=1) + df['genre_random']) > 2).astype(int)
df['liked_values'] = ((df[relevant_features].sum(axis=1) + df['genre_random']))

X = df[numerical_features].values
y = df['liked'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.2f}")

df['predicted_like_prob'] = gnb.predict_proba(X)[:, 1]

recommendations = df.sort_values(by='predicted_like_prob', ascending=False).head(10)

print("\nRecommended songs based on your preferences:")
for _, row in recommendations.iterrows():
    print(
        f"Track: {row['track_name']} by {row['artists']} | "
        f"Liked Value: {row['liked_values']:.2f} | "
        f"Sum of Relevant Features: {row[relevant_features].sum():.2f} | "
    )