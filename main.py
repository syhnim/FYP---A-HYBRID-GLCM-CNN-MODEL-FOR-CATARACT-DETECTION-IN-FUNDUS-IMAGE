import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras import Model
from keras.applications import ResNet50
from keras.layers import Flatten
from keras.models import load_model
import cv2
from skimage.feature import graycomatrix, graycoprops
import pickle

# GLCM
distance = 1
angles = [0, 45, 90, 135]

# Initialize the selected file name variable
selected_file_name = ""

# Load pre-trained model
model = load_model('C:/Users/ACER/PycharmProjects/FYP/Cataract Disease/vgg19_model.h5')
model1 = load_model('C:/Users/ACER/PycharmProjects/FYP/Cataract Disease/GLCM-CNN Final Model.h5')  # Replace with the path to your saved model file
# Load the k-NN model from the pickle file
with open('C:/Users/ACER/PycharmProjects/FYP/knn_model.pkl', 'rb') as model2:
    knn_model = pickle.load(model2)


# CNN
# Function to preprocess the image for testing
def preprocess_imageCNN(image_path):
    img = Image.open(image_path)
    img = img.resize((112, 112))  # Resize the image to fit the model's input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    return img

# GLCM-k-NN
# Function to preprocess the image for testing
def preprocess_imageGLCMkNN(image_path):
    img = Image.open(image_path)
    img = img.resize((480, 640))  # Resize the image to fit the model's input size
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Normalize pixel values to the range [0, 1]
    img = cv2.equalizeHist(img)

    # Calculate GLCM matrix
    glcm = graycomatrix(img, distances=[distance], angles=angles, levels=256, symmetric=True, normed=True)

    # Calculate GLCM properties (contrast, homogeneity, energy, correlation)
    glcm_props = ['contrast', 'homogeneity', 'energy', 'correlation']

    # Initialize an empty list to store GLCM features
    features = []

    for prop in glcm_props:
        glcm_feature = graycoprops(glcm, prop).ravel()
        if glcm_feature.size == 1:
            # GLCM feature extraction failed for this property, set to 0
            glcm_feature = np.array([0])
        features.extend(glcm_feature)

    # Flatten both GLCM and ResNet-50 features to 1D arrays
    glcm_features_flat = np.array(features)

    return glcm_features_flat


# GLCM-CNN
# Function to preprocess the image for testing
def preprocess_imageGLCMCNN(image_path):
    # Load the image using PIL
    pil_img = Image.open(image_path)
    pil_img = pil_img.resize((256, 256))  # Resize the image to fit the model's input size
    img = np.array(pil_img)  # Convert PIL image to NumPy array
    img = cv2.medianBlur(img, 5)

    # Resnet-50 feature extraction
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Add a Flatten layer to the ResNet-50 model
    flatten_layer = Flatten()(resnet_model.layers[-1].output)

    # Create a new model that includes the Flatten layer
    feature_extraction_model = Model(inputs=resnet_model.input, outputs=flatten_layer)

    # Preprocess and extract ResNet-50 features for the image
    resnet_features = feature_extraction_model.predict(
        img[np.newaxis, ...])  # Add np.newaxis to create a batch of size 1

    # GLCM feature extraction
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate GLCM matrix
    glcm = graycomatrix(gray_image, distances=[distance], angles=angles, levels=256, symmetric=True, normed=True)

    # Calculate GLCM properties (contrast, dissimilarity, homogeneity, energy, correlation)
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    # Initialize an empty list to store GLCM features
    features = []

    for prop in glcm_props:
        glcm_feature = graycoprops(glcm, prop).ravel()
        if glcm_feature.size == 1:
            # GLCM feature extraction failed for this property, set to 0
            glcm_feature = np.array([0])
        features.extend(glcm_feature)

    # Flatten both GLCM and ResNet-50 features to 1D arrays
    glcm_features_flat = np.array(features)
    resnet_features_flat = resnet_features.flatten()

    # Min-max scaling for GLCM features
    min_val = np.min(glcm_features_flat)
    max_val = np.max(glcm_features_flat)
    glcm_features_scaled = (glcm_features_flat - min_val) / (max_val - min_val)

    # Combine GLCM and ResNet-50 features
    combined_features = np.hstack((glcm_features_scaled, resnet_features_flat))

    return combined_features


# Function to handle image upload and prediction
def upload_and_predict():
    global selected_file_name  # Use the global variable
    selected_file_name = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])

    if selected_file_name:
        file_name_label.config(text=f"Selected File: {selected_file_name}")  # Update the file name label

        # Load and display the selected image
        img_pil = Image.open(selected_file_name)
        img_pil = img_pil.resize((300, 300))  # Resize the image to fit the GUI
        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Resize the image frame to match the uploaded image size
        image_frame.config(width=img_tk.width(), height=img_tk.height())

        # Preprocess the image for the CNN model
        img_cnn = preprocess_imageCNN(selected_file_name)
        img_cnn = np.expand_dims(img_cnn, axis=0)

        # Preprocess the image for the GLCM-CNN model
        img_glcm_cnn = preprocess_imageGLCMCNN(selected_file_name)
        img_glcm_cnn = np.expand_dims(img_glcm_cnn, axis=0)

        # Preprocess the image for the GLCM-CNN model
        img_glcm_knn = preprocess_imageGLCMkNN(selected_file_name)
        img_glcm_knn = np.expand_dims(img_glcm_knn, axis=0)

        # Map numeric labels to class names
        class_names = {0: "Normal", 1: "Cataract"}

        # Make predictions using both the CNN and GLCM-CNN models
        predicted_class_name_cnn = class_names[model.predict(img_cnn).argmax()]

        # Predict for GLCM-CNN model
        prediction_glcm_cnn = model1.predict(img_glcm_cnn)
        if prediction_glcm_cnn > 0.5:
            predicted_class_name_glcm_cnn = "Cataract"
        else:
            predicted_class_name_glcm_cnn = "Normal"

        # Predict for GLCM-k-NN model
        predicted_class_glcm_knn = knn_model.predict(img_glcm_knn)

        result_label.config(
            text=f"CNN Predicted Class: {predicted_class_name_cnn}\n"
                 f"GLCM-CNN Predicted Class: {predicted_class_name_glcm_cnn}\n"
                 f"GLCM-k-NN Predicted Class: {predicted_class_glcm_knn[0]}\n")

# Function to reset the displayed image and prediction result
def reset_image():
    global selected_file_name  # Use the global variable
    selected_file_name = ""  # Reset the selected file name
    file_name_label.config(text="")  # Clear the file name label
    image_label.config(image=None)  # Clear the displayed image
    result_label.config(text="Predicted Class: ")

# Create the main GUI window
root = tk.Tk()
root.title("Cataract Detection System")  # Set the title

# Create GUI components
title_label = tk.Label(root, text="Cataract Detection System", font=("Helvetica", 16))
title_label.pack(pady=10)  # Add padding to the top of the label

# Create a frame for displaying the image (box)
image_frame = tk.Frame(root, borderwidth=2, relief="groove")
image_frame.pack(pady=10)  # Add padding below the title

# Create a label to display the selected file name
file_name_label = tk.Label(root, text="", font=("Helvetica", 12))
file_name_label.pack(pady=10)  # Add padding between the file name label and the upload button

upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_button.pack(pady=10)  # Add padding between the upload button and the result label

reset_button = tk.Button(root, text="Reset", command=reset_image)  # Add reset button
reset_button.pack(pady=10)  # Add padding below the reset button

image_label = tk.Label(image_frame)
image_label.pack()

result_label = tk.Label(root, text="\n\nPredicted Class: ")
result_label.pack()

# Start the GUI event loop
root.mainloop()