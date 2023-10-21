import os
import matplotlib.pyplot as plt
import numpy as np
# cv2 comes from opencv-python
import cv2
import tensorflow as tf

# https://www.youtube.com/watch?v=bte8Er0QhDg

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model= tf.keras.models.Sequential()

# # transforma a imagem 28x28 em 784
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# # * Entender o que é Relu e Activation Function com detalhes
# # 128 neuronios pra processar cada padrao
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # 128 neuronios pra processar cada padrao
# # 128 neuronios pra processar cada padrao
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # 0 1 2 2 3 4 5 6 7 8 9
# # Softmax all the outputs add up to 1
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # epochs = quantas vezes o codigo vai ver os mesmos training datas
# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.model')


# loading so we do not have to re train it over and over again
model = tf.keras.models.load_model('handwritten.model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

image_index = 0

while os.path.isfile(f"digits\digit{image_index}.png"):
    try:
        img = cv2.imread(f"digits\digit{image_index}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("oof error")
    finally:
        image_index += 1
