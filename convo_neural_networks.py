import tensorflow as tf
from utils import predict

from tensorflow import keras
import matplotlib.pyplot as plt

# Load and Split Dataset
(train_images,
 train_labels), (test_images,
                 test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'sheep', 'truck'
]

# # View images
# index = 6
# plt.imshow(train_images[index])
# plt.xlabel(class_names[train_labels[index][0]])
# plt.show()

# Build the convolutional base
model = keras.models.Sequential()
model.add(
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

# summary = model.summary()
# print(summary)

# Add dense layers (classifier)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10))

# summary = model.summary()
# print(summary)

# compile and train the model
# compile
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# train
trained_model = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

# Evaluate the model against the test images to see the % accuracy
test_loss,test_acc = model.evaluate(test_images, test_labels, verbose=2)

# make predictions
prediction = model.predict(test_images)

# view prediction
index = 5
label_index = test_labels[index][0]


predict(class_names, test_images, label_index, prediction, index)



# TODO - Using a Pre0trained Model
