import tensorflow as tf
import sys
import os

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout


# === DATASET LOADING AND PREPARATION ===

# Load MNIST Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the Pixels
x_train, x_test = x_train/255, x_test/255

# Convert output from integer labels (0-9) to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Reshape to 4D to be compatible with CNNs (batch_size, height, width, channels)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)


# === BUILDING THE NEURAL NETWORK ===

# Create convolutional neural network
model = keras.Sequential([
    # Convolutional Layer

    # Learn 32 filters using 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # Max-pooling Layer: Pool the max of each 2x2 grid in the image
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units into one single layer
    tf.keras.layers.Flatten(),


    # Neural Network Layer

    # Add a hidden layer with 50% dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add output layer with output units for all 10 digits
    tf.keras.layers.Dense(10, activation="softmax")
])


# === COMPILE THE CONVOLUTIONAL NEURAL NETWORK

# Compile and train the CNN
model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=10)

# Evaluate the performance of CNN
model.evaluate(x_test, y_test, verbose=2)


# === SAVE THE MODEL TO A FILE FOR REUSABILITY ===

# Define filename
filename = "model.h5"

# Check if the file exists, and remove it before saving a new model
if os.path.exists(filename):
    os.remove(filename)
    print(f"Existing model {filename} deleted.")

# Save the new model
model.save(filename)
print(f"New model saved to {filename}.")
