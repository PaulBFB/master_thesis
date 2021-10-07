from tensorflow.keras import models, layers, Model
from tensorflow.keras.optimizers import Adam


def make_model(
    input_shape: tuple = (11, ),
    number_hidden_layers: int = 8, 
    activation: str = 'relu', 
    neurons: int = 32,
    loss: str = 'binary_crossentropy',
    optimizer: str = 'adam',
    learning_rate: float = .003
) -> Model:
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
    
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


if __name__ == '__main__':
    test_model = make_model()
    
    test_model.summary()
