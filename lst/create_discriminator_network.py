import tensorflow as tf


def create_discriminator_network(
    number_hidden_layers: int = 2,
    number_hidden_units_power: int = 5,
    hidden_activation_function: str = 'LeakyReLU',
    use_dropout: bool = True,
    upsampling: bool = True,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.3,
    number_output_units: int = 1) -> tf.keras.Model:

    model = tf.keras.Sequential()

    for i in range(number_hidden_layers):
        
        if upsampling:
            # implements the guideline - downsample in the discriminator network
            model.add(tf.keras.layers.Dense(2 ** (number_hidden_units_power + number_hidden_layers - i - 1)))
        
        else:
            model.add(tf.keras.layers.Dense(2 ** number_hidden_units_power))
        
        if use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        else:
            pass
            
        model.add(tf.keras.layers.Activation(hidden_activation_function))
        
        if use_dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            pass

    model.add(tf.keras.layers.Dense(number_output_units, activation=None))

    return model
