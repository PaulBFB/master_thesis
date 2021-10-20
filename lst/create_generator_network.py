import tensorflow as tf


def create_generator_network(
    number_hidden_layers: int = 2,
    number_hidden_units_power: int = 5,
    hidden_activation_function: str = 'ReLU',
    use_dropout: bool = False,
    upsampling: bool = True,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.3,
    number_output_units: int = 12,
    output_activation_function: str = 'tanh') -> tf.keras.Model:

    model = tf.keras.Sequential()
    for i in range(number_hidden_layers):
        
        if upsampling:
            # implements the guideline DCGAN - upsampling layers in the generator
            model.add(tf.keras.layers.Dense(2 ** (number_hidden_units_power + i), use_bias=False))
        
        else:
            model.add(tf.keras.layers.Dense(2 ** number_hidden_units_power, use_bias=False))
        
        if use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        else:
            pass
        
        model.add(tf.keras.layers.Activation(hidden_activation_function))
        
        if use_dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            pass

    model.add(tf.keras.layers.Dense(number_output_units))
    model.add(tf.keras.layers.Activation(output_activation_function))

    return model
