import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, UpSampling2D, Input, Concatenate
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split
import os

# Load dataset
def load_images(path, size=(128, 128)):
    images = []
    for filename in os.listdir(path):
        img = load_img(os.path.join(path, filename), target_size=size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# Data preparation
def preprocess_images(images):
    # Convert RGB images to LAB color space
    images = images.astype('float32') / 255.0
    lab_images = rgb2lab(images)
    X = lab_images[..., 0] / 100.0  # L channel (lightness)
    Y = lab_images[..., 1:] / 128.0  # AB channels (color)
    return X[..., np.newaxis], Y

# Load dataset
dataset_path = 'path_to_images_directory'
images = load_images(dataset_path)
X, Y = preprocess_images(images)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# CNN model definition
def build_colorization_model():
    input_layer = Input(shape=(128, 128, 1))

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(conv1)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(conv2)

    # Decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv5)

    output_layer = Conv2D(2, (3, 3), activation='tanh', padding='same')(up2)

    return Model(inputs=input_layer, outputs=output_layer)

# Compile model
model = build_colorization_model()
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=16, epochs=50)

# Predict on test data
predictions = model.predict(X_test)

# Convert predictions to RGB
def postprocess_output(X, predictions):
    predictions = predictions * 128.0
    X = X.squeeze() * 100.0
    colorized_images = []
    for i in range(len(predictions)):
        lab_image = np.zeros((128, 128, 3))
        lab_image[..., 0] = X[i]
        lab_image[..., 1:] = predictions[i]
        colorized_images.append(lab2rgb(lab_image))
    return np.array(colorized_images)

colorized_images = postprocess_output(X_test, predictions)

# Display original grayscale and colorized images
for i in range(5):
    plt.figure(figsize=(10, 5))

    # Original grayscale
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Grayscale')

    # Colorized
    plt.subplot(1, 2, 2)
    plt.imshow(colorized_images[i])
    plt.title('Colorized')

    plt.show()
