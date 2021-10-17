import tensorflow as tf
import numpy as np


#generator = tf.keras.models.load_model('./models/best_generator.h5', compile=False)
tf.random.set_seed(42)
noise_generator = tf.random.Generator.from_seed(42)


def generate_data(
    model: tf.keras.models.Model,
    latent_space_shape: int = 8,
    number_samples: int = 50):
    
    noise = noise_generator.normal(shape=(number_samples, latent_space_shape))

    data = model(noise, training=False)
    x_train = data[:, :-1].numpy()
    y_train = data[:, -1].numpy()
    y_train = np.abs(np.where(y_train < 0, 0, 1))
    
    result = {
        'x_train': x_train,
        'y_train': y_train}

    return result


if __name__ == '__main__':
    
    test_data = generate_data(number_samples=50)
    print(test_data)
#    np.save('./data/titanic_generated.npy', test_data)
