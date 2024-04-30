import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the model and data
model = load_model('trained_model.h5')
X_train_cnn = np.load('X_train_cnn.npy')
y_train_cnn = np.load('y_train_cnn.npy')
X_test_cnn = np.load('X_test_cnn.npy')
y_test_cnn = np.load('y_test_cnn.npy')

# Concatenate the datasets for PCA
X_combined = np.concatenate((X_train_cnn, X_test_cnn))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined.reshape(X_combined.shape[0], -1))  # Flatten images for scaling

# Perform PCA
pca = PCA(n_components=2)  # Reduce to two dimensions for visualization
principal_components = pca.fit_transform(X_scaled)

# Visualization
plt.figure(figsize=(10, 8))
# Use a single color, e.g., 'blue', for all data points
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], color='blue', alpha=0.5)
plt.title('PCA of Breakhis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
