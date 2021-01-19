import numpy as np
import matplotlib.pyplot as plt


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