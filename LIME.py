import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the datasets from .npy files
# Assuming you have already loaded the model and datasets
model = load_model('trained_model.h5')
X_test_cnn = np.load('X_test_cnn.npy')
y_test_cnn = np.load('y_test_cnn.npy')
filenames_test = np.load('filenames_test.npy', allow_pickle=True)

# Select images to explain
indices_to_analyze = [i for i, filename in enumerate(filenames_test) if filename in filenames_test[:5]]
images_to_analyze = X_test_cnn[indices_to_analyze]


# Preprocess images if necessary (e.g., normalization used during training)
def preprocess_input(x):
    return x.astype('float32') / 255.0  # Adjust based on actual preprocessing used

# Creating a prediction function that LIME can use
def predict(model, img_array):
    img_array = preprocess_input(img_array)  # Preprocess the image
    return model.predict(img_array)

# Initialize LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

# Choose a specific image to explain
image_idx = 0
image_to_explain = images_to_analyze[image_idx]

# Explanation with LIME
explanation = explainer.explain_instance(image_to_explain.astype('double'), 
                                         classifier_fn=lambda x: predict(model, x), 
                                         top_labels=5, hide_color=0, num_samples=1000)

# Display the original image and the explanation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image_to_explain)
ax1.set_title('Original Image')

# Get image and mask for LIME explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
ax2.imshow(mark_boundaries(temp / 2 + 0.5, mask))
ax2.set_title('LIME Explanation')
plt.show()