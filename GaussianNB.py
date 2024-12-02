import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('dataset.csv')

unique_genres = sorted(set(df['track_genre']))
genre_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
df['genre_int'] = df['track_genre'].map(genre_mapping)

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence']

scaler = StandardScaler()
music = df.dropna(subset=numerical_features)
df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))

X = df[numerical_features].values
y = df['genre_int']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_val)

report = classification_report(y_val, y_pred, target_names=unique_genres)
print("Classification Report:\n", report)

cm = confusion_matrix(y_val, y_pred)

cm_flattened = cm.flatten()
top_two_values = np.sort(cm_flattened)[-3:]
top_three_indices = cm_flattened.argsort()[-3:][::-1] 
row_idx, col_idx = np.unravel_index(top_three_indices, cm.shape) 
top_genres = [(unique_genres[row_idx[i]], unique_genres[col_idx[i]], cm[row_idx[i], col_idx[i]]) for i in range(3)]

print(f"Top 3 highest values in the confusion matrix:")
for genre1, genre2, value in top_genres:
    print(f"True Genre: {genre1}, Predicted Genre: {genre2}, Value: {value}")

plt.figure(figsize=(10, 7))
sns.heatmap(cm, fmt="d", cmap="Blues", xticklabels=unique_genres, yticklabels=unique_genres)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.title('Confusion Matrix')
plt.show()
