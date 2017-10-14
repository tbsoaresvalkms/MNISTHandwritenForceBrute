import numpy as np
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam, TFOptimizer

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

    return (X_train, y_train), (X_test, y_test)


def neural_network(batch_size, epochs, hidden_layer, hidden_neuro, dropout, optimizer, activation):
    (X_train, y_train), (X_test, y_test) = get_data()

    model = Sequential()
    model.add(Dense(hidden_neuro, input_shape=(X_train.shape[1],), activation=activation))
    model.add(Dropout(dropout))
    for i in range(hidden_layer):
        model.add(Dense(hidden_neuro, activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2)

    return model.evaluate(X_test, y_test, verbose=2)


batch_size = [32, 128, 256, 1024]
epochs = [20, 50, 100, 200]
hidden_layer = [0, 1, 3]
hidden_neuro = [8, 32, 64, 128]
dropout = [0., 0.1, 0.2]
optimizer = [SGD(), Adam(), RMSprop()]
activations = ['relu', 'tanh', 'sigmoid']

scores = []

for b in batch_size:
    for e in epochs:
        for hl in hidden_layer:
            for hn in hidden_neuro:
                for d in dropout:
                    for o in optimizer:
                        for a in activations:
                            params = (b, e, hl, hn, d, o, a)
                            print('\nStart: ', params)
                            score = neural_network(batch_size=params[0], epochs=params[1], hidden_layer=params[2],
                                                   hidden_neuro=params[3], dropout=params[4], optimizer=params[5],
                                                   activation=params[6])
                            scores.append((params, score[1]))
                            with open('status_neural_network', 'a') as file:
                                file.write(str((params, score[1])) + '\n')
