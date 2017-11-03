import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

np.random.seed(1671)


def get_data():
    # X_train = 60000, 28, 28 / y_train = 60000
    # X_test = 10000, 28, 28 / y_test = 10000
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

    return X_train, y_train, X_test, y_test


def neural_network():
    model = Sequential()
    model.add(Dense(128, input_shape=(784,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model


X_train, Y_train, X_test, Y_test = get_data()

classifier = KerasClassifier(build_fn=neural_network, epochs=150, batch_size=32, verbose=2)
results = cross_val_score(classifier, X_train, Y_train, cv=10, n_jobs=10)

print(results.mean())
