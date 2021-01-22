import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def predict(class_names, test_images, label_index, prediction_model, index):

    predicted = np.argmax(prediction_model[index])

    print(f"Expected {class_names[label_index]}")
    print(f"Guessed {class_names[predicted]}")

    if predicted == label_index:
        print('Good Guess!, welldone Model')
    else:
        print('Oops Wrong, Lets try again')

    plt.figure(facecolor='b', edgecolor='r', clear=True)
    plt.imshow(test_images[index])
    plt.show()
    plt.close()


# https://keras.io/api/datasets/imdb/#getwordindex-function
word_index = keras.datasets.imdb.get_word_index()


# encode input string to array of vocabulary keys
def encode(text, MAXLEN):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return keras.preprocessing.sequence.pad_sequences([tokens],
                                                      maxlen=MAXLEN)[0]
