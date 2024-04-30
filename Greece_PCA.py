import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def preprocess_images(folder_path):
    data = []
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        return np.array(data)  # Return an empty numpy array if the path does not exist

    file_list = os.listdir(folder_path)
    if not file_list:
        print(f"No files found in directory: {folder_path}")
        return np.array(data)  # Return an empty numpy array if no files in the directory

    for filename in file_list:
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # Resize to unify image sizes
            img_array = np.array(img) / 255.0  # Normalize pixels to 0-1
            data.append(img_array.flatten())  # Flatten the 3D array to 1D
        else:
            print(f"Skipped non-PNG file: {filename}")

    if not data:
        print("No PNG images processed.")
    return np.array(data)

# Path to the image folder
greece_image_folder = r"C:\Users\shrip\Desktop\BITS_sem8\CS_F407_AI\Assignment\Greece"

# Preprocess the images
data = preprocess_images(greece_image_folder)

if data.size == 0:
    print("No data to process for PCA.")
else:
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    principal_components = pca.fit_transform(X_scaled)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title('PCA of Images from Greece')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
