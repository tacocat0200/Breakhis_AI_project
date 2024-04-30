import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef, mean_squared_error, cohen_kappa_score, precision_recall_curve, auc, brier_score_loss


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

# Flatten the image data for traditional ML models
X_flat = all_data.reshape(all_data.shape[0], -1)  # This changes the shape from (n, 224, 224, 3) to (n, 150528)

# Split the flat data
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    X_flat, all_labels, test_size=0.2, random_state=42
)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_flat, y_train_flat)
rf_predictions = rf_clf.predict(X_test_flat)
print("Random Forest Classifier:")
print(classification_report(y_test_flat, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test_flat, rf_predictions))

# SVM Classifier
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train_flat, y_train_flat)
svm_predictions = svm_clf.predict(X_test_flat)
print("Support Vector Machine (SVM):")
print(classification_report(y_test_flat, svm_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test_flat, svm_predictions))


# KNN Classifier
# knn_clf = KNeighborsClassifier(n_neighbors=5)
# knn_clf.fit(X_train_flat, y_train_flat)
# knn_predictions = knn_clf.predict(X_test_flat)
# print("K-Nearest Neighbors (KNN):")
# print(classification_report(y_test_flat, knn_predictions))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test_flat, knn_predictions))

# # # Save SVM outputs
# # np.save('svm_predictions.npy', svm_predictions)
# # np.save('svm_test_labels.npy', y_test_flat)

# # # Save KNN outputs
# # np.save('knn_predictions.npy', knn_predictions)
# # np.save('knn_test_labels.npy', y_test_flat)

# #------extra metrics----

# # Train and evaluate classifiers
# classifiers = [
#     (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
#     (SVC(kernel='linear', probability=True, random_state=42), "SVM"),
#     (KNeighborsClassifier(n_neighbors=5), "KNN")
# ]

# for clf, name in classifiers:
#     clf.fit(X_train_flat, y_train_flat)
#     predictions = clf.predict(X_test_flat)
#     probabilities = clf.predict_proba(X_test_flat)[:, 1] if hasattr(clf, "predict_proba") else None
#     print(f"{name} Classifier:")
#     print(classification_report(y_test_flat, predictions))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test_flat, predictions))

#     # Calculate additional metrics
#     tn, fp, fn, tp = confusion_matrix(y_test_flat, predictions).ravel()
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     prevalence = np.mean(y_test_flat)
#     roc_auc = roc_auc_score(y_test_flat, probabilities) if probabilities is not None else "N/A"
#     precision, recall, _ = precision_recall_curve(y_test_flat, probabilities) if probabilities is not None else (None, None, None)
#     pr_auc = auc(recall, precision) if probabilities is not None else "N/A"
#     mcc = matthews_corrcoef(y_test_flat, predictions)
#     mse = mean_squared_error(y_test_flat, predictions)
#     kappa = cohen_kappa_score(y_test_flat, predictions)
#     brier_score = brier_score_loss(y_test_flat, probabilities) if probabilities is not None else "N/A"

#     print(f"Sensitivity: {sensitivity}")
#     print(f"Specificity: {specificity}")
#     print(f"Prevalence: {prevalence}")
#     print(f"AUC-ROC: {roc_auc}")
#     print(f"AUC-PR: {pr_auc}")
#     print(f"MCC: {mcc}")
#     print(f"MSE: {mse}")
#     print(f"Cohen’s Kappa: {kappa}")
#     print(f"Brier Score: {brier_score}")