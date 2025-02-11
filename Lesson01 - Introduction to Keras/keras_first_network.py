import tensorflow as tf

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


# === LOAD DATASET
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split the dataset into input (x) and output (y)
x = dataset[:, 0:8]
y = dataset[:, 8]


# === DEFINE KERAS MODEL
model = Sequential()

# Explicitly add Input Layer
model.add(Input(shape=(8,)))  # This replaces input_shape in Dense

# Add hidden layer 01
model.add(Dense(12, activation='relu'))
# model.add(Dropout(0.2))

# Add hidden layer 02
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))

# Add output layer
model.add(Dense(1, activation='sigmoid'))


# == COMPILE, TRAIN, AND EVALUATE KERAS MODEL
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Keras model
model.fit(x, y, epochs=150, batch_size=10)

# Evaluate the Keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy * 100))
