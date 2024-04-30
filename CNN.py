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
import shap # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, mean_squared_error, cohen_kappa_score, brier_score_loss
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef, mean_squared_error, cohen_kappa_score, precision_recall_curve



def create_output_folder(output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

def resize_image(input_path, output_path, target_size=(224, 224)):
    try:
        img = Image.open(input_path)
        img_resized = img.resize(target_size)
        img_resized.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def preprocess_images(folder_path, label):
    data = []
    labels = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img_array = np.array(img) / 255.0
            data.append(img_array)
            labels.append(label)
            filenames.append(filename)

    return data, labels, filenames

# Paths (you can write your own paths)
benign_image_folder = r"C:\Users\shrip\Desktop\BITS_sem8\CS_F407_AI\Assignment\Breakhis\benign"
malignant_image_folder = r"C:\Users\shrip\Desktop\BITS_sem8\CS_F407_AI\Assignment\Breakhis\malignant"
resized_benign_folder = r"C:\Users\shrip\Desktop\BITS_sem8\CS_F407_AI\Assignment\Breakhis\resized_images\benign"
resized_malignant_folder = r"C:\Users\shrip\Desktop\BITS_sem8\CS_F407_AI\Assignment\Breakhis\resized_images\malignant"

# Create output folders
create_output_folder(resized_benign_folder)
create_output_folder(resized_malignant_folder)

# Resize images
for folder_path, output_folder_path in [(benign_image_folder, resized_benign_folder),
                                        (malignant_image_folder, resized_malignant_folder)]:
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder_path, filename)
        resize_image(input_path, output_path)


# Load and preprocess images
benign_data, benign_labels, benign_filenames = preprocess_images(resized_benign_folder, 0)
malignant_data, malignant_labels, malignant_filenames = preprocess_images(resized_malignant_folder, 1)

# Combine data and labels
all_data = np.array(benign_data + malignant_data)
all_labels = np.array(benign_labels + malignant_labels)
all_filenames = np.array(benign_filenames + malignant_filenames)

# Shuffle the data
all_data, all_labels, all_filenames = shuffle(all_data, all_labels, all_filenames, random_state=42)

# Display attributes for the first 5 images
for filename, label in zip(all_filenames[:5], all_labels[:5]):
    print(f"pic_name: {filename}, label: {label}")

# Prepare the data for the neural network
X_cnn = all_data.reshape(-1, 224, 224, 3)
y_cnn = all_labels

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, filenames_train, filenames_test = train_test_split(
    X_cnn, y_cnn, all_filenames, test_size=0.2, random_state=42
)

for filename, label in zip(filenames_train[:5], y_train_cnn[:5]):
    print(f"Training set - Filename: {filename}, Label: {label}")

for filename, label in zip(filenames_test[:5], y_test_cnn[:5]):
    print(f"Testing set - Filename: {filename}, Label: {label}")


np.save('X_train_cnn.npy', X_train_cnn)
np.save('X_test_cnn.npy', X_test_cnn)
np.save('y_train_cnn.npy', y_train_cnn)
np.save('y_test_cnn.npy', y_test_cnn)
np.save('filenames_train.npy', filenames_train)
np.save('filenames_test.npy', filenames_test)



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Using sigmoid since it's likely a binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cnn)
print("Test Accuracy:", test_accuracy)

# Predict classes
predictions = model.predict(X_test_cnn)
predicted_classes = (predictions > 0.5).astype(int)

model.save('trained_model.h5')  # H5 format


# Confusion Matrix
conf_matrix = confusion_matrix(y_test_cnn, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

# # Calculate Sensitivity and Specificity
# tn, fp, fn, tp = confusion_matrix(y_test_cnn, predicted_classes).ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)

# # Prevalence
# prevalence = np.mean(y_test_cnn)

# # AUC-ROC and AUC-PR
# roc_auc = roc_auc_score(y_test_cnn, predictions)
# precision, recall, _ = precision_recall_curve(y_test_cnn, predictions)
# pr_auc = auc(recall, precision)

# # Matthews Correlation Coefficient
# mcc = matthews_corrcoef(y_test_cnn, predicted_classes)

# # Mean Squared Error
# mse = mean_squared_error(y_test_cnn, predictions)

# # Print metrics
# print(f"Sensitivity: {sensitivity}")
# print(f"Specificity: {specificity}")
# print(f"Prevalence: {prevalence}")
# print(f"AUC-ROC: {roc_auc}")
# print(f"AUC-PR: {pr_auc}")
# print(f"MCC: {mcc}")
# print(f"MSE: {mse}")

# # Additional Metrics
# # Fowlkes-Mallows - typically used in clustering scenarios but can be adapted if relevant
# if 'knn_predictions' in locals():
#     fm_index = stats.fowlkes_mallows_score(y_test_cnn, knn_predictions)
#     print(f"Fowlkes-Mallows Index: {fm_index}")

# # Cohen’s Kappa
# kappa = cohen_kappa_score(y_test_cnn, predicted_classes)
# print(f"Cohen’s Kappa: {kappa}")

# # Gini Coefficient - assuming predictions are probabilities
# gini_coefficient = 2 * roc_auc - 1
# print(f"Gini Coefficient: {gini_coefficient}")

# # Brier Score
# brier_score = brier_score_loss(y_test_cnn, predictions)
# print(f"Brier Score: {brier_score}")

# Classification Report
print(classification_report(y_test_cnn, predicted_classes))

# Correctly classified filenames
correct_indices = np.where(predicted_classes.flatten() == y_test_cnn)[0]
incorrect_indices = np.where(predicted_classes.flatten() != y_test_cnn)[0]

correct_filenames = filenames_test[correct_indices]
incorrect_filenames = filenames_test[incorrect_indices]

np.save('correct_filenames.npy', correct_filenames)
np.save('incorrect_filenames.npy', incorrect_filenames)

print("Correctly Classified Filenames:")
for filename in correct_filenames[:5]:
    print(filename)

print("Misclassified Filenames:")
for filename in incorrect_filenames[:5]:
    print(filename)
