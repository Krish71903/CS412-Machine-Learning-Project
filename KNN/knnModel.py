import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset into a pandas dataframe
music = pd.read_csv('dataset.csv')

# Create a mapping of genres to integer labels for multiclass classification
unique_genres = sorted(set(music['track_genre']))
genre_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
music['genre_int'] = music['track_genre'].map(genre_mapping)

# Define the list of numerical features to use for model training
numerical_features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence'
]

# Standardize the numerical features using StandardScaler to have mean=0 and variance=1
scaler = StandardScaler()
music = music.dropna(subset=numerical_features)  # Drop rows with missing values in numerical features
music[numerical_features] = scaler.fit_transform(music[numerical_features].fillna(0))  # Apply scaling and fill empty sections with 0

# Prepare the features
X = music[numerical_features].values
y = music['genre_int']

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply PCA to reduce the dimensionality while preserving 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Output the original and reduced dimensions of the feature set
print(f"Original dimensions: {X_train.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")

# Train a K-Nearest Neighbors classifier with 3 neighbors and Euclidean distance metric
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train_pca, y_train)

# Predict the labels on the test set
y_pred = knn.predict(X_test_pca)

# Output the classification report and model accuracy
report = classification_report(y_test, y_pred, target_names=unique_genres)
print("Test Score:", knn.score(X_test_pca, y_test))
print(report)

# Compute the confusion matrix to evaluate the performance of the classifier
cm = confusion_matrix(y_test, y_pred)

# Flatten the confusion matrix to identify the top 3 highest values
cm_flattened = cm.flatten()
top_two_values = np.sort(cm_flattened)[-3:]  # Get the top 3 values
top_three_indices = cm_flattened.argsort()[-3:][::-1]  # Get the indices of the top 3 values
row_idx, col_idx = np.unravel_index(top_three_indices, cm.shape)  # Get row and column indices of the top values
top_genres = [(unique_genres[row_idx[i]], unique_genres[col_idx[i]], cm[row_idx[i], col_idx[i]]) for i in range(3)]

# Print the top 3 highest values in the confusion matrix along with their genres
print(f"Top 3 highest values in the confusion matrix:")
for genre1, genre2, value in top_genres:
    print(f"True Genre: {genre1}, Predicted Genre: {genre2}, Value: {value}")

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, fmt="d", cmap="Blues", xticklabels=unique_genres, yticklabels=unique_genres)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.title('Confusion Matrix')

# Show the confusion matrix plot
plt.show()
