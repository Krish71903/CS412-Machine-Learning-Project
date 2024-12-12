import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error


df = pd.read_csv('dataset.csv')

np.random.seed(42) 
unique_genres = df['track_genre'].unique()
unique_artists = df['artists'].unique()
unique_album_name = df['album_name'].unique()


track_genre_random_values = {genre: np.random.uniform(0, 1) for genre in unique_genres}
artists_random_values = {artist: np.random.uniform(0, 1) for artist in unique_artists}
album_name_random_values = {album: np.random.uniform(0, 1) for album in unique_album_name}

relevant_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

df['track_genre_random'] = df['track_genre'].map(track_genre_random_values)
df['artists_random'] = df['artists'].map(artists_random_values)
df['album_name_random'] = df['album_name'].map(album_name_random_values)

df['liked_values'] = ((df[relevant_features].sum(axis=1) + df['track_genre_random'] + df['artists_random'] + df['album_name_random']))
df['liked'] = ((df['liked_values']) > 4.5).astype(int)

num_liked = df[df['liked'] == 1].shape[0]
num_disliked = df[df['liked'] == 0].shape[0]

print(f"Number of liked songs (liked = 1): {num_liked}")
print(f"Number of disliked songs (liked = 0): {num_disliked}")

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))


X = df[numerical_features].values
y = df['liked'].values

X_reshaped = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)


model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(32, kernel_size=2, activation='sigmoid', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=2, activation='sigmoid', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))


evaluation = model.evaluate(X_val, y_val)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")


y_pred = model.predict(X_val).ravel()
y_pred_classes = (y_pred > 0.5).astype(int)  

# Calculate precision and recall
precision = precision_score(y_val, y_pred_classes)
recall = recall_score(y_val, y_pred_classes)


rmse = np.sqrt(mean_squared_error(y_val, y_pred))


df['predicted_like'] = model.predict(X_reshaped)


recommendations = df.sort_values(by='predicted_like', ascending=False).head(10)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"RMSE: {rmse}")


print("\nRecommended songs based on your preferences:")
for _, row in recommendations.iterrows():
    print(f"Track: {row['track_name']} by {row['artists']} {row['liked']} {row['liked_values']}")