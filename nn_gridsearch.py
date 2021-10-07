import os
from tensorflow.keras import models, layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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
    
    optimizer = Adam(lr=learning_rate)
    
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


def logdir(hyperparam_note=None) -> str:
    
    
    run_d = time.strftime(
        f'run_%Y_%m_%d-%H_%M_%S{"_" + hyperparam_note if hyperparam_note is not None else ""}')
    
    directory = os.path.join(root_logdir, run_d)
    
    return directory
    

if __name__ == '__main__':
    test_model = make_model()
    print(logdir('changed_batch_size_32'))

    
    test_model.summary()
