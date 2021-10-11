import tensorflow as tf
import numpy as np


generator = tf.keras.models.load_model('./models/generator_latest.h5')


def generate_data(
    model: tf.keras.models.Model = generator,
    number_samples: int = 50):
    
    noise = tf.random.normal(shape=(50, 12))

    data = generator(noise, training=False)
    
    result = {
        'x_train': data[:, :-1],
        'y_train': data[:, -1]}

    return result


if __name__ == '__main__':
    
    test_data = generate_data(number_samples=50)
#    np.save('./data/titanic_generated.npy', test_data)
