import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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


def neural_network(output_shape, input_shape, hidden_neuro, hidden_layer, optimizer, init, dropout):
    model = Sequential()
    model.add(Dense(hidden_neuro, input_shape=input_shape, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(hidden_layer):
        model.add(Dense(hidden_neuro / 2, kernel_initializer=init, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(output_shape, kernel_initializer=init, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


X_train, Y_train, X_test, Y_test = get_data()

batch_size = [32, 256, 1024]
epochs = [20, 50, 200]
hidden_layer = [0, 1, 3]
hidden_neuro = [64, 128]
dropout = [0.2]
optimizer = ['adam', 'rmsprop']
init = ['glorot_uniform', 'normal', 'uniform']
input_shape = [(784,)]
output_shape = [(10)]

classifier = KerasClassifier(build_fn=neural_network, verbose=2)

early = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=0, mode='auto')
callbacks_list = [early]

params = dict(batch_size=batch_size,
              epochs=epochs,
              hidden_layer=hidden_layer,
              hidden_neuro=hidden_neuro,
              dropout=dropout,
              optimizer=optimizer,
              init=init,
              input_shape=input_shape,
              output_shape=output_shape)

grid = GridSearchCV(estimator=classifier, param_grid=params, cv=10, n_jobs=10)
grid_result = grid.fit(X_train, Y_train, callbacks=callbacks_list, validation_data=(X_test, Y_test))

line = "Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
best_model = grid_result.best_estimator_.model

for mean, stdev, param in zip(means, stds, params):
    line += "%f (%f) with: %r\n" % (mean, stdev, param)

with open('data/grid_result_ANN', 'w') as file:
    file.write(line)
with open('data/model_ANN.json', 'w') as file:
    file.write(best_model.to_json())

best_model.save_weights('data/weights_ANN.h5')
