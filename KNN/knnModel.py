import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

music = pd.read_csv('dataset.csv')

unique_genres = sorted(set(music['track_genre']))
genre_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
music['genre_int'] = music['track_genre'].map(genre_mapping)

numerical_features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence'
]

scaler = StandardScaler()
music = music.dropna(subset=numerical_features)
music[numerical_features] = scaler.fit_transform(music[numerical_features].fillna(0))

X = music[numerical_features].values
y = music['genre_int']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original dimensions: {X_train.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)

report = classification_report(y_test, y_pred, target_names=unique_genres)
print("Test Score:", knn.score(X_test_pca, y_test))
print(report)

cm = confusion_matrix(y_test, y_pred)

cm_flattened = cm.flatten()
top_two_values = np.sort(cm_flattened)[-3:]
top_three_indices = cm_flattened.argsort()[-3:][::-1]
row_idx, col_idx = np.unravel_index(top_three_indices, cm.shape)
top_genres = [(unique_genres[row_idx[i]], unique_genres[col_idx[i]], cm[row_idx[i], col_idx[i]]) for i in range(3)]

print(f"Top 3 highest values in the confusion matrix:")
for genre1, genre2, value in top_genres:
    print(f"True Genre: {genre1}, Predicted Genre: {genre2}, Value: {value}")

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, fmt="d", cmap="Blues", xticklabels=unique_genres, yticklabels=unique_genres)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.title('Confusion Matrix')
plt.show()