import pandas as pd # type: ignore # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from keras import layers # type: ignore

df = pd.read_csv('dataset.csv')

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))

relevant_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

df[relevant_features] = scaler.fit_transform(df[relevant_features].fillna(0))


np.random.seed(42) 
unique_genres = df['track_genre'].unique()
genre_random_values = {genre: np.random.rand() for genre in unique_genres}

df['genre_random'] = df['track_genre'].map(genre_random_values)

df['liked'] = ((df[relevant_features].sum(axis=1) + df['genre_random']) > 2).astype(int)
df['liked_values'] = ((df[relevant_features].sum(axis=1) + df['genre_random']))

X = df[numerical_features].values
y = df['liked'].values

X_reshaped = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)), 
    layers.Conv1D(32, kernel_size=2, activation='relu', padding='same'), 
    layers.MaxPooling1D(pool_size=2), 
    layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'), 
    layers.GlobalMaxPooling1D(),  
    layers.Dense(64, activation='relu'),  
    layers.Dropout(0.5), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

evaluation = model.evaluate(X_val, y_val)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")

df['predicted_like'] = model.predict(X_reshaped).flatten()

recommendations = df.sort_values(by='predicted_like', ascending=False).head(10)

print("\nRecommended songs based on your preferences:")
for _, row in recommendations.iterrows():
    print(
        f"Track: {row['track_name']} by {row['artists']} | "
        f"Liked Value: {row['liked_values']} | "
        f"Sum of Relevant Features: {row[relevant_features].sum():.2f} | "
        f"Random Genre Value: {row['genre_random']:.2f}"
    )
