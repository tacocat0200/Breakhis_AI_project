import shap  # Make sure SHAP is installed and imported
import numpy as np
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model
# Assuming 'model' is your trained model and 'X_test_cnn' and 'y_test_cnn' are your test datasets.
# 'filenames_test' should be an array of filenames corresponding to 'X_test_cnn'.

class_labels = ['benign', 'malignant']
mapping = {0: 'benign', 1: 'malignant'}

# Load the datasets from .npy files
X_train_cnn = np.load('X_train_cnn.npy')
X_test_cnn = np.load('X_test_cnn.npy')
y_train_cnn = np.load('y_train_cnn.npy')
y_test_cnn = np.load('y_test_cnn.npy')
filenames_train = np.load('filenames_train.npy', allow_pickle=True)
filenames_test = np.load('filenames_test.npy', allow_pickle=True)

# Load the previously saved model
model = load_model('trained_model.h5')

filenames_test = np.load('filenames_test.npy', allow_pickle=True)

# Filenames to analyze
filenames_to_analyze = filenames_test[:10]

# Find indices of these filenames in the test set
indices_to_analyze = [i for i, filename in enumerate(filenames_test) if filename in filenames_to_analyze]

# Fetch the images and labels corresponding to these indices
images_to_analyze = X_test_cnn[indices_to_analyze]
labels_to_analyze = y_test_cnn[indices_to_analyze]

masker = shap.maskers.Image("blur(28,28)", images_to_analyze[0].shape)

# Create the SHAP explainer instance using the model and the masker
explainer = shap.Explainer(model, masker)

# Compute SHAP values
shap_values = explainer(images_to_analyze)

np.save('shap_values_50.npy', shap_values.values)
print("SHAP values saved successfully.")


# Load SHAP values
shap_values = np.load('shap_values_50.npy')

# Load corresponding image data
X_test_cnn = np.load('X_test_cnn.npy')  # Ensure this points to the correct dataset

# Suppose you want to visualize a specific subset or all images
# Here's how to select a specific subset, e.g., first 10, or a random selection
number_to_visualize = 10  # Adjust this number based on how many you want to visualize

# You can use slicing to select a continuous range
indices_to_analyze = range(number_to_visualize)  # E.g., first 10 images

# Or use np.random.choice to randomly select a number of indices
# indices_to_analyze = np.random.choice(len(X_test_cnn), number_to_visualize, replace=False)

images_to_plot = X_test_cnn[indices_to_analyze]

# Correct SHAP values shape if needed
if shap_values.shape[-1] == 1:
    shap_values = shap_values[..., 0]

print("SHAP values shape:", shap_values.shape)
print("Images to plot shape:", images_to_plot.shape)

# Attempt to plot SHAP values
try:
    # Plot a subset or all SHAP values
    shap.image_plot(shap_values[indices_to_analyze], images_to_plot)
except Exception as e:
    print("Error plotting SHAP values:", e)

# If you need to plot in batches due to memory constraints
batch_size = 10  # Define an appropriate batch size
for i in range(0, len(indices_to_analyze), batch_size):
    batch_indices = indices_to_analyze[i:i+batch_size]
    batch_images = X_test_cnn[batch_indices]
    batch_shap_values = shap_values[batch_indices]
    try:
        shap.image_plot(batch_shap_values, batch_images)
    except Exception as e:
        print(f"Error plotting SHAP values for batch starting at index {i}:", e)


