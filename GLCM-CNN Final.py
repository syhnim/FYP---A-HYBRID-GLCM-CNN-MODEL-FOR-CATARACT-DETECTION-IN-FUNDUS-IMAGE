import glob
import os
import cv2
import numpy as np
from category_encoders import OneHotEncoder
from keras.utils import to_categorical
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import pickle
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, concatenate, Input, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Initialize variables
data_images = []
data_labels = []
imSize = 256  # Set your desired image size here
distance = 1
angles = [0, 45, 90, 135]
#glcm_props = ["contrast", "homogeneity", "energy", "correlation"]
# Create a label encoder to convert labels to numerical values
label_encoder = LabelEncoder()
num_classes = 2
# Map labels to numeric values, matching the folder names exactly
label_map = {"Cataract": 1, "Normal": 0}

# STEP 1: Load and preprocess the dataset
for directory_path in glob.glob("C:/Users/ACER/PycharmProjects/FYP/Cataract Disease/*"):
    label = directory_path.split("\\")[-1]
    if label == "normal":
        label = 0
    elif label == "cataract":
        label = 1

    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        fundIm = cv2.imread(img_path)
        resizeIm = cv2.resize(fundIm, (imSize, imSize))
        noiseIm = cv2.medianBlur(resizeIm, 5)
        data_images.append(noiseIm)
        data_labels.append(label_map[label])

# Convert lists to numpy arrays
train_images = np.array(data_images)
train_labels = np.array(data_labels)

print(train_labels[0])

# Encode class labels as integers
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Save the extracted GLCM features to a pickle file
with open('data_label.pkl', 'wb') as file:
    pickle.dump(train_labels_encoded, file)

print("GLCM features have been saved to 'glcm_features.pkl'.")

from skimage.feature import graycomatrix, graycoprops
# STEP 2: Function to extract GLCM features for a given image
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [distance], angles, 256, symmetric=True, normed=True)
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = [graycoprops(glcm, prop).ravel() for prop in glcm_props]
    return np.hstack(features)

# Extract GLCM features for all images (assuming data_images is a list of images)
glcm_features = [extract_glcm_features(image) for image in data_images]

# Convert the GLCM features to a numpy array and normalize them
glcm_features = np.array(glcm_features)
glcm_features = (glcm_features - glcm_features.mean(axis=0)) / glcm_features.std(axis=0)

# Example: Get the shape of the first feature vector in glcm_features
shape_of_first_feature_vector = glcm_features[0].shape
print(shape_of_first_feature_vector)
print(glcm_features.shape)

# Save the extracted GLCM features to a pickle file
with open('glcm_features.pkl', 'wb') as file:
    pickle.dump(glcm_features, file)

print("GLCM features have been saved to 'glcm_features.pkl'.")


import csv

# Assuming you have glcm_features as a list of feature vectors
# Each feature vector represents GLCM features for an image

# Define the CSV file path
csv_file_path = 'glcm_features 1.csv'

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row with feature names
    header = ['Image'] + [f'GLCM_{i}' for i in range(len(glcm_features[0]))]
    writer.writerow(header)

    # Write the data rows
    for i, feature_vector in enumerate(glcm_features):
        row = [f'Image_{i + 1}'] + list(map(str, feature_vector))
        writer.writerow(row)

print(f'GLCM features have been saved to {csv_file_path}.')


# STEP 3: Feature Extraction by using Resnet-50

# Load ResNet-50 model with pre-trained weights (exclude top classification layer)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(imSize, imSize, 3))

# Add a Flatten layer to the ResNet-50 model
flatten_layer = Flatten()(resnet_model.layers[-1].output)

# Create a new model that includes the Flatten layer
feature_extraction_model = Model(inputs=resnet_model.input, outputs=flatten_layer)

# Preprocess and extract ResNet-50 features for all images
resnet_features = feature_extraction_model.predict(train_images)

# Flatten the 4D array into a 2D matrix
resnet_features_flat = resnet_features.reshape(resnet_features.shape[0], -1)

# Save the flattened ResNet-50 features to a file using pickle
with open('resnet_features_vector.pkl', 'wb') as file:
    pickle.dump(resnet_features_flat, file)

print("ResNet-50 features as a 2D vector have been saved.")

print("2D shape Resnet-50:",resnet_features_flat.shape)

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the pre-extracted GLCM features and ResNet-50 features
with open('glcm_features.pkl', 'rb') as glcm_file:
    glcm_features = pickle.load(glcm_file)

with open('resnet_features_vector.pkl', 'rb') as resnet_file:
    resnet_features_flat = pickle.load(resnet_file)

with open('data_label.pkl', 'rb') as label_file:
    data_labels_binary = pickle.load(label_file)

print("Label shape:", data_labels_binary.shape)


# STEP 4: Classification by using DNN

# Combine GLCM and ResNet-50 features
combined_features = np.hstack((glcm_features, resnet_features_flat))

# Verify the shape of combined_features
print("Shape of combined_features:", combined_features.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, data_labels_binary, test_size=0.3, random_state=42)

# Define your DNN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(combined_features.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model with binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary() # summary of the model

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluate the model on the test data
y_pred_binary = model.predict(X_test)
y_pred_classes_binary = (y_pred_binary > 0.5).astype(int)  # Convert probabilities to binary classes (0 or 1)
accuracy_binary = accuracy_score(y_test, y_pred_classes_binary)

print(f"Test Accuracy (Binary Classification): {accuracy_binary * 100:.2f}%")

model.save('GLCM-CNN Final Model.h5')


# STEP 5: Performance Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix for binary classification
conf_matrix_binary = confusion_matrix(y_test, y_pred_classes_binary)

class_names = ["normal", "cataract"]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall (sensitivity), and F1-score
precision = precision_score(y_test, y_pred_classes_binary)
recall = recall_score(y_test, y_pred_classes_binary)
f1 = f1_score(y_test, y_pred_classes_binary)

# Print the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Calculate the classification report
report = classification_report(y_test, y_pred_classes_binary, target_names=class_names)

# Print the classification report
print("Classification Report:")
print(report)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()








