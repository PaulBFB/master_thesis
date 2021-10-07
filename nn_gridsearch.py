import os
import time
from tensorflow.keras import models, layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import RandomizedSearchCV


root_logdir = os.path.join(os.curdir, 'custom_logs')


def make_model(
    input_shape: tuple = (11, ),
    number_hidden_layers: int = 8, 
    activation: str = 'relu', 
    neurons: int = 32,
    loss: str = 'binary_crossentropy',
    learning_rate: float = .003) -> Model:
    """
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for i in range(number_hidden_layers):
        model.add(layers.Dense(
            neurons, 
            name=f'hidden_layer_{i}', 
            activation='relu'
        ))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


def logdir(hyperparam_note=None) -> str:
    
    
    run_d = time.strftime(
        f'run_%Y_%m_%d-%H_%M_%S{"_" + hyperparam_note if hyperparam_note is not None else ""}')
    
    directory = os.path.join(root_logdir, run_d)
    
    return directory


def nn_gridsearch(
    model_builder_function,
    x_train,
    y_train,
    params,
    epochs: int = 100,
#    validation_split: float = .2,
    patience: int = 10,
    checkpoints: bool = True):
    
    
#    validation_length = int(x_train.shape[1] * validation_split)   
    
#    x_val =  x_train[:validation_length]
#    x_train = x_train[validation_length:]
    
#    y_val =  y_train[:validation_length]
#    y_train = y_train[validation_length:]
    
    
    model_wrapped = KerasClassifier(
        model_builder_function,
        batch_size=32,
        shuffle=True,
        verbose=1)
    
    
    gridsearch = RandomizedSearchCV(model_wrapped, params, n_iter=10, cv=3)
    gridsearch.fit(
        x_train, y_train,
        epochs=epochs,
#        validation_data=(x_val, y_val),
        callbacks=[TensorBoard(logdir())])
    
    return gridsearch


if __name__ == '__main__':
    from process_data import process_data
    
    
    data = process_data()
    
    test_model = make_model()
    print(logdir('changed_batch_size_32'))
    
#    test_model.summary()
    
    grid_parameters = {'number_hidden_layers': [3, 5, 7]}
    
    # note = the wrapper takes a FUNCTION as input!
    grid = nn_gridsearch(
        make_model, 
        data['x_train_processed'], data['y_train'], 
        grid_parameters)

    
    best_model = grid.best_estimator_.model
    best_model.save('./models/titanic_gridsearch.h5')
    
    print(best_model.summary())
