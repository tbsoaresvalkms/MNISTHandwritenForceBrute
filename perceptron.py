import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

np.random.seed(1671)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_train = X_train.astype('float32')
X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')

# normalize data
X_train /= 255
X_test /= 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
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

