import tensorflow as tf
import numpy as np
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist.load_data()
(train_images, train_label), (test_images, test_label) = fashion_mnist

class_names = [
    'T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]

# # set some figure props. figure = image + background
# f = plt.figure(facecolor='b', edgecolor='r', clear=True)
# # draws the image
# plt.imshow(train_images[9])
# # displays it
# plt.show()
# # clean up memory
# plt.close()


# PRE PROCESS DATA. VALUES SHOULD BE BETWEEN 0 AND 1
train_images = train_images / 255.0
test_images = test_images / 255.0


# CREATING THE MODEL
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2) //relu = REctified Linear Unit
    keras.layers.Dense(10, activation='softmax')  # output layer (3)
])

# COMPILE() - CONFIGURES THE MODEL FOR TRAINING
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# FIT() - TRAINS THE MODEL FOR A FIXED NUMBER OF EPOCHS
model.fit(train_images, train_label, epochs=4)


# RETURNS THE METRICS FOR THE MODEL IN TEST MODE (using actual test data)
test_loss, test_acc = model.evaluate(test_images, test_label, verbose=1)


# PREDICT() - GENERATES OUTPUT PREDICTION FOR INPUT SAMPLES
predictions = model.predict(test_images)


# TEST PREDICTIONS
# pick an index in test_images to predict
guess = 789

predicted = np.argmax(predictions[guess])
expected = test_label[guess]


print(f"Expected {class_names[test_label[guess]]}")
print(f"Guessed {class_names[predicted]}")

if predicted == expected:
    print('Good Guess!, welldone Model')
else:
    print('Oops Wrong, Lets try again')

f = plt.figure(facecolor='b', edgecolor='r', clear=True)
plt.imshow(test_images[guess])
plt.show()
plt.close()