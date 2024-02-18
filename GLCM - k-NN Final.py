# GLCM - k-NN model

import glob
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import pickle

# Initialize variables
train_images = []
train_labels = []
imSizeWidth = 480
imSizeHeight = 640
distance = 1
angles = [0, 45, 90, 135]
glcm_props = ["contrast", "homogeneity", "energy", "correlation"]

# Load and preprocess the dataset
for directory_path in glob.glob("C:/Users/ACER/PycharmProjects/FYP/Cataract Disease/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        fundIm = cv2.imread(img_path)
        resizeIm = cv2.resize(fundIm, (imSizeWidth, imSizeHeight))
        # convert RGB -> greyscale
        grayIm = cv2.cvtColor(resizeIm, cv2.COLOR_BGR2GRAY)
        # histogram equalisation
        equIm = cv2.equalizeHist(grayIm)
        train_images.append(grayIm)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Function to extract GLCM features for a given image
def extract_glcm_features(image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(image, [distance], angles, 256, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop).ravel() for prop in glcm_props]
    return np.hstack(features)

# Extract GLCM features for all images
glcm_features = [extract_glcm_features(image) for image in train_images]

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(glcm_features, train_labels, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(glcm_features, train_labels, test_size=0.2, random_state=42)

# Specify the values of k you want to test
k_values = [3, 5, 7, 9]

# Plot both training and testing accuracy for specific k values
training_accuracies = []
testing_accuracies = []


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    sensitivity = recall_score(y_test, y_pred, average='weighted')

    print(f'For k = {k}:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Sensitivity (Recall): {sensitivity:.4f}')

    # Print classification report
    class_report = classification_report(y_test, y_pred)
    print(f'Classification Report (k = {k}):\n{class_report}')

    # Generate and plot the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (k = {k})')
    plt.colorbar()

    # Annotate the confusion matrix cells with counts
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, str(confusion_mat[i, j]), ha='center', va='center')

    tick_marks = np.arange(len(sorted(np.unique(y_train))))
    # Limit the number of tick labels for better readability
    subset_labels = sorted(np.unique(y_train))[:10]  # Adjust the number as needed
    plt.xticks(tick_marks[:10], subset_labels, rotation=45)
    plt.yticks(tick_marks, subset_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #plt.savefig('confusion_matrix glcm-knn.png')

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Training accuracy
    y_train_pred = knn.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    training_accuracies.append(training_accuracy)

    # Testing accuracy
    y_test_pred = knn.predict(X_test)
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    testing_accuracies.append(testing_accuracy)

    # Calculate accuracy, precision, and sensitivity (recall)
    accuracy = accuracy_score(y_test, y_pred)

plt.plot(k_values, training_accuracies, marker='o', label='Training Accuracy')
plt.plot(k_values, testing_accuracies, marker='o', label='Testing Accuracy')
plt.title('Training and Testing Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
plt.savefig('graph accuracy glcm-knn.png')
plt.show()


# Save the trained model to a pickle file
model_filename = "knn_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(knn, model_file)

# Load the model from the pickle file
loaded_knn_model = None
with open(model_filename, "rb") as model_file:
    loaded_knn_model = pickle.load(model_file)

# Make predictions on new data (replace the ellipsis with GLCM features of new data)
# For example, you can use the test set again as new data to demonstrate making predictions
new_data_glcm_features = X_test
predictions = loaded_knn_model.predict(new_data_glcm_features)
print("Predictions on new data:", predictions)

from sklearn.metrics import precision_score, recall_score

# Calculate precision and sensitivity (recall)
precision = precision_score(y_test, y_pred, average='weighted')
sensitivity = recall_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Sensitivity (Recall):", sensitivity)






