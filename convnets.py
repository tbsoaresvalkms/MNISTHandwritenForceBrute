import numpy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import utils
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

numpy.random.seed(1671)

# X_train = 60000, 28, 28 / y_train = 60000
# X_test = 10000, 28, 28 / y_test = 10000
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:, :, :, numpy.newaxis].astype('float32')
X_test = X_test[:, :, :, numpy.newaxis].astype('float32')

# normalize data
X_train /= 255
X_test /= 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, padding='same', kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, batch_size=256, epochs=10, validation_split=0.2)
score = model.evaluate(X_test, y_test)

print('\nTest score: ', score[0])
print('Test accuracy: ', score[1])
print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
