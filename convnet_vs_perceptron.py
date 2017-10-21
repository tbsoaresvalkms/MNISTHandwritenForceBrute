import numpy
import matplotlib.pyplot as plt
from keras import utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

data_amount = 60000

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:data_amount].astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train_convnet = X_train[:, :, :, numpy.newaxis]
X_test_convnet = X_test[:, :, :, numpy.newaxis]
X_train_perceptron = X_train.reshape(X_train.shape[0], 784)
X_test_perceptron = X_test.reshape(X_test.shape[0], 784)

y_train = utils.to_categorical(y_train[:data_amount], 10)
y_test = utils.to_categorical(y_test, 10)

model_convnet = Sequential()
model_convnet.add(Conv2D(32, padding='same', kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model_convnet.add(MaxPooling2D(pool_size=(2, 2)))
model_convnet.add(Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
model_convnet.add(MaxPooling2D(pool_size=(2, 2)))
model_convnet.add(Flatten())
model_convnet.add(Dense(500, activation='relu'))
model_convnet.add(Dense(10, activation='softmax'))
model_convnet.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model_convnet.summary()

history_convnet = model_convnet.fit(X_train_convnet, y_train, batch_size=64, epochs=10, validation_split=0.2)

model_perceptron = Sequential()
model_perceptron.add(Dense(128, input_shape=(784,), activation='relu'))
model_perceptron.add(Dropout(0.3))
model_perceptron.add(Dense(128, activation='relu'))
model_perceptron.add(Dropout(0.3))
model_perceptron.add(Dense(64, activation='relu'))
model_perceptron.add(Dropout(0.3))
model_perceptron.add(Dense(10, activation='softmax'))
model_perceptron.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model_perceptron.summary()

history_perceptron = model_perceptron.fit(X_train_perceptron, y_train, batch_size=64, epochs=10, validation_split=0.2)

plt.plot(history_convnet.history['acc'])
plt.plot(history_convnet.history['val_acc'])
plt.plot(history_perceptron.history['acc'])
plt.plot(history_perceptron.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_convet', 'test_convet', 'train_perceptron', 'test_perceptron'], loc='upper left')
plt.show()
