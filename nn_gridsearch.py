import os
import time
import numpy as np
from scipy.stats import reciprocal
from tensorflow.keras import models, layers, Model
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from process_data import process_data


root_logdir = os.path.join(os.curdir, 'custom_logs')


def make_model(
    input_shape: tuple = (11, ),
    number_hidden_layers: int = 8, 
    activation: str = 'elu', 
    alpha: float = .2,
    neurons: int = 32,
    loss: str = 'binary_crossentropy',
    learning_rate: float = .003,
    dropout_rate: float = .5) -> Model:
    """
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for i in range(number_hidden_layers):
        model.add(layers.Dense(
            neurons, 
            kernel_initializer='he_normal',
            name=f'hidden_layer_{i}_relu_alpha_{alpha}'))
        
        if number_hidden_layers >= 3:
            model.add(layers.BatchNormalization())
        
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i}_{round(dropout_rate * 100)}'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', Precision(), Recall(), AUC()])
    
    return model


def logdir(hyperparam_note=None) -> str:
    
    
    run_d = time.strftime(
        f'run_%Y_%m_%d-%H_%M_%S{"_" + hyperparam_note if hyperparam_note is not None else ""}')
    
    directory = os.path.join(root_logdir, run_d)
    
    return directory


def nn_gridsearch(
    make_model_function,
    x_train: np.ndarray = None,
    y_train: np.ndarray = None,
    params: dict = None,
    epochs: int = 100,
    validation_split: float = .1,
    patience: int = 10,
    batch_size: int = 16,
    n_iterations: int = 10,
    verbose: int = 1):
    
    keras_cl = KerasClassifier(
        make_model_function,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose)
    
    rnd_search_cv = RandomizedSearchCV(
        keras_cl, 
        params, 
        n_iter=n_iterations, 
        cv=3, 
        verbose=2, 
        n_jobs=-1)
    
    rnd_search_cv.fit(
        x_train, y_train, 
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
#            EarlyStopping(patience=patience, monitor='val_loss', mode='min'),
            TensorBoard(logdir())])
    
    return rnd_search_cv


if __name__ == '__main__':
    from process_data import process_data
    
    
    data = process_data()
    
    test_model = make_model()
#    print(logdir('changed_batch_size_32'))
        
    grid_parameters = {
        'number_hidden_layers': list(range(1, 8)),
        'neurons': np.arange(1, 100).tolist(),
        'learning_rate': reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
        'dropout_rate': np.arange(.2, .6, .1).tolist(),
        'alpha': np.arange(.2, .35, .05).tolist(),
        'activation': ['elu', 'selu', 'relu']
    }

    
    # note = the wrapper takes a FUNCTION as input!
    grid = nn_gridsearch(
        make_model, 
        data['x_train_processed'], data['y_train'],
        grid_parameters,
        n_iterations=1)

    
    best_model = grid.best_estimator_.model
    best_model.save(f'./models/titanic_gridsearch_{time.strftime("%Y_%m_%d_%H_%M")}.h5')
    
    print(best_model.summary())
