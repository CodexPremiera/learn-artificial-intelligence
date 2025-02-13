# 📝 Handwritten Digit Recognition with CNN and Pygame

This project is a **handwritten digit recognition system** that uses a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. Users can draw a digit on a 28x28 grid using Pygame, and the trained model will classify the digit in real time.

## 📌 Features
- **Interactive Drawing Grid:** Users can draw digits on a 28x28 grid using the mouse.
- **Real-time Classification:** The CNN model predicts the drawn digit when the user clicks the "Classify" button.
- **Reset Functionality:** Users can clear the grid and draw a new digit.
- **Model Training & Saving:** The trained CNN model is saved as `model.h5` for future predictions.

## 🏗️ Installation & Setup
### 🔹 Prerequisites
Ensure you have **Python 3.x** installed along with the following dependencies:

```
pip install tensorflow numpy pygame
```

### 🔹 Running the Project
#### 1️⃣ Train the Model (if needed)
If you want to retrain the model, run:
```sh
python handwriting.py
```
This will train a CNN model on the MNIST dataset and save it as `model.h5`.

#### 2️⃣ Run the Digit Recognition Game
```sh
python recognition.py
```
This launches the Pygame interface where you can draw and classify digits.

## 🎮 How to Use
1. **Run the game** using `python recognition.py`.
2. **Draw a digit (0-9)** on the 28x28 grid using the mouse.
3. Click **"Classify"** to predict the digit.
4. Click **"Reset"** to clear the grid and draw again.
5. The **predicted digit** will be displayed on the right side of the screen.

## ⚙️ Model Architecture
The CNN model is structured as follows:
- **Conv2D Layer:** 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D:** 2x2 pool size
- **Flatten Layer:** Converts 2D feature maps into a single vector
- **Dense Layer:** 128 neurons, ReLU activation, Dropout (50%)
- **Output Layer:** 10 neurons (for digits 0-9), Softmax activation

## 🏆 Acknowledgments
This project is referenced from the **Harvard CS50 Introduction to Artificial Intelligence** lecture on Neural Networks.

## 📜 License
MIT License. Feel free to modify and improve the project!

---
Happy Coding! 🚀

