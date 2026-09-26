# MNIST Digit Classification Using Deep Learning


📌 Project Overview

This project focuses on handwritten digit classification using Deep Learning.

The model is trained using the MNIST dataset, which contains images of handwritten digits from 0 to 9. A neural network is used to learn patterns from these images and predict the correct digit.



🎯 Goal: Build a Deep Learning model that can recognize handwritten digits with high accuracy.



🧠 About MNIST Dataset

MNIST stands for Modified National Institute of Standards and Technology.

The dataset contains:

🖼️ 60,000 training images

🧪 10,000 testing images

🔢 Digits from 0 to 9

📐 Image size: 28 × 28 pixels

⚫ Grayscale images



Each image represents one handwritten digit.




Example:

0  1  2  3  4  5  6  7  8  9



🛠️ Technologies Used

🐍 Python

🧠 TensorFlow / Keras

📊 NumPy

📈 Matplotlib

🔬 Scikit-learn

📓 Jupyter Notebook / Google Colab

📂 Project Structure



MNIST-Digit-Classification/

│

├── 📓 MNIST_Digit_Classification.ipynb

├── 📄 README.md

├── 📁 dataset/

├── 📁 images/

│   └── sample_predictions.png

└── 📄 requirements.txt



🚀 Project Workflow

MNIST Dataset

      ↓

Data Loading

      ↓
Data Preprocessing

      ↓
Normalization

      ↓
Build Neural Network

      ↓
Train Model

      ↓
Evaluate Model

      ↓
Predict Digits

      ↓
Display Results



🔧 Step 1: Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


📥 Step 2: Load MNIST Dataset
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)




🔄 Step 3: Data Preprocessing

The pixel values range from 0 to 255.

We normalize them between 0 and 1.

X_train = X_train / 255.0
X_test = X_test / 255.0

Normalization helps the neural network learn more efficiently.



🧠 Step 4: Build Neural Network

A simple neural network can be created using Keras.

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
Model Architecture
28 × 28 Image

     ↓
Flatten Layer

     ↓
Dense Layer – 128 Neurons

     ↓
ReLU Activation

     ↓
Output Layer – 10 Neurons

     ↓
Softmax

     ↓
Predicted Digit (0–9)




⚙️ Step 5: Compile the Model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    
)


🏋️ Step 6: Train the Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.1
)



📊 Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)

The accuracy may vary depending on the model architecture and training settings.




🔮 Step 8: Make Predictions
predictions = model.predict(X_test)

predicted_digit = np.argmax(predictions[0])

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])
🖼️ Display Prediction
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()




📈 Results


The model learns to classify handwritten digits into 10 classes (0–9).

Performance Metrics
Metric	Result
Training Samples	60,000
Testing Samples	10,000
Classes	10
Image Size	28 × 28
Model	Neural Network
Optimizer	Adam
Activation	ReLU + Softmax

Add your actual accuracy after training the model.




Example:

Test Accuracy: XX.XX%




💡 Key Learning Outcomes

Through this project, I learned:

✅ Basics of Deep Learning

✅ Neural Network architecture

✅ MNIST dataset handling

✅ Image preprocessing

✅ Data normalization

✅ Model training

✅ Model evaluation

✅ Digit prediction

✅ TensorFlow and Keras



🔮 Future Improvements


The project can be improved by:

🚀 Using a Convolutional Neural Network (CNN)

📈 Improving model accuracy

🎨 Building a web interface for handwritten digit input

📱 Creating a mobile application

☁️ Deploying the model online

✍️ Allowing users to draw a digit and predict it




📌 Conclusion


This project demonstrates how Deep Learning can be used for handwritten digit recognition.

The MNIST dataset provides a simple and effective way to understand the complete Machine Learning/Deep Learning workflow—from data preprocessing to model training, evaluation, and prediction.




👩‍💻 Author

Alfiya Inamdar

🎓 Artificial Intelligence & Data Science Student
