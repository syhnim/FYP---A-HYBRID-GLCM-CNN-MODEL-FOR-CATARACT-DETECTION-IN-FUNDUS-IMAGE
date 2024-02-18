# cnn model- the final model

from keras.applications import VGG19
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from matplotlib import pyplot as plt
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import classification_report, plot_confusion_matrix


def create_pretrained_vgg19(num_classes):
    base_model = VGG19(include_top=False, weights='imagenet', input_tensor=Input(shape=(112, 112, 3)))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dense(4096, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Replace num_classes with the number of classes in your classification task
num_classes = 2  # For example
model = create_pretrained_vgg19(num_classes)

model.summary()

import os
import glob
import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical


# Load and preprocess the dataset
train_images = []
train_labels = []

imSize = 112

label_encoder = LabelEncoder()

# Map labels to numeric values, matching the folder names exactly
label_map = {"Cataract": 1, "Normal": 0}

# Load and preprocess the dataset
for directory_path in glob.glob("C:/Users/ACER/PycharmProjects/FYP/Cataract Disease/*"):
    label = directory_path.split("\\")[-1]

    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        fundIm = cv2.imread(img_path)
        resizeIm = cv2.resize(fundIm, (imSize, imSize))
        train_images.append(resizeIm)
        train_labels.append(label_map[label])  # Map label to numeric value

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

print("Train label:", train_labels[0])

# Normalize RGB values to [0, 1]
train_images = train_images.astype('float32') / 255.0

# Split the dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

print("Train image: ", train_images.shape)
print("Train label", train_labels.shape)

# Create the VGG-19 model
model = create_pretrained_vgg19(num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 100
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save('vgg19_model.h5')

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Model Accuracy CNN.png')

plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Model Loss CNN.png')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Predict labels on the test set
test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
confusion = confusion_matrix(np.argmax(test_labels, axis=1), test_predictions)

# Define class names
class_names = ["Normal", "Cataract"]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Convert test_predictions to one-hot encoding
test_predictions_onehot = to_categorical(test_predictions, num_classes=num_classes)

# Calculate the classification report
report = classification_report(test_labels, test_predictions_onehot, target_names=class_names)

# Print the classification report
print("Classification Report:")
print(report)



