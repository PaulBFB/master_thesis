import tensorflow as tf
import numpy as np
from process_data import process_data
import time


tf.random.set_seed(42)
np.random.seed(42)

if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'

    
data = process_data()    
x_train = data['x_train_processed']
y_train = data['y_train']

x_train_tf = tf.data.Dataset.from_tensor_slices(x_train)
x_train_tf = x_train_tf.map(preprocess)


def create_generator_network(
    number_hidden_layers: int = 1,
    number_hidden_units: int = 100,
    number_output_units: int = 784) -> tf.keras.Model:

    model = tf.keras.Sequential()
    for i in range(number_hidden_layers):
        model.add(tf.keras.layers.Dense(number_hidden_units, use_bias=False))
        model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(number_output_units, activation='tanh'))

    return model


def create_discriminator_network(
    number_hidden_layers: int = 1,
    number_hidden_units: int = 100,
    number_output_units: int = 1) -> tf.keras.Model:

    model = tf.keras.Sequential()

    for i in range(number_hidden_layers):
        model.add(tf.keras.layers.Dense(number_hidden_units,))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=.5))

    model.add(tf.keras.layers.Dense(number_output_units, activation=None))


    return model


@tf.function
def preprocess(
    passenger: np.ndarray,
    output_shape: tuple = (11,),
    mode: str='uniform'):
    
    assert passenger.shape == output_shape, f'data shape needs to match the expected output shape {output_shape} - shape is {passenger.shape}'
    
    passenger = tf.convert_to_tensor(passenger)
    
    if mode == 'uniform':
        input_z = tf.random.uniform(
            shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        input_z = tf.random.normal(shape=(z_size,))
    return input_z, passenger



data_shape = (11,)
z_size = 11 # dimension of the latent space
mode_z = 'uniform'
    
num_epochs = 100
batch_size = 64
image_size = (28, 28)

generator_hidden_layers = 1
generator_hidden_units = 100

discriminator_hidden_layers = 1
discriminator_hidden_units = 100


if mode_z == 'uniform':
    fixed_z = tf.random.uniform(
        shape=(batch_size, z_size),
        minval=-1, maxval=1)
elif mode_z == 'normal':
    fixed_z = tf.random.normal(
        shape=(batch_size, z_size))


def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))    
    return (images+1)/2.0

## Set-up the dataset
mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(
    lambda ex: preprocess(ex, mode=mode_z))

mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(
    batch_size, drop_remainder=True)

## Set-up the model
with tf.device(device_name):
    generator_model = create_generator_network(
        number_hidden_layers=generator_hidden_layers,
        number_hidden_units=generator_hidden_units,
        number_output_units=np.product(data_shape))
    
    generator_model.build(input_shape=(None, z_size))
    
    discriminator_model = create_discriminator_network(
        number_hidden_layers=discriminator_hidden_layers,
        number_hidden_units=discriminator_hidden_units)
    
    discriminator_model.build(input_shape=(None, np.prod(data_shape)))

## Loss function and optimizers:
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

all_losses = []
all_d_vals = []
epoch_samples = []

start_time = time.time()
for epoch in range(1, num_epochs+1):
    epoch_losses, epoch_d_vals = [], []
    for i,(input_z,input_real) in enumerate(mnist_trainset):
        
        ## Compute generator's loss
        with tf.GradientTape() as g_tape:
            g_output = generator_model(input_z)
            d_logits_fake = discriminator_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)
            
        g_grads = g_tape.gradient(g_loss, generator_model.trainable_variables)
        g_optimizer.apply_gradients(
            grads_and_vars=zip(g_grads, generator_model.trainable_variables))

        ## Compute discriminator's loss
        with tf.GradientTape() as d_tape:
            d_logits_real = discriminator_model(input_real, training=True)

            d_labels_real = tf.ones_like(d_logits_real)
            
            d_loss_real = loss_fn(
                y_true=d_labels_real, y_pred=d_logits_real)

            d_logits_fake = discriminator_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)

            d_loss_fake = loss_fn(
                y_true=d_labels_fake, y_pred=d_logits_fake)

            d_loss = d_loss_real + d_loss_fake

        ## Compute the gradients of d_loss
        d_grads = d_tape.gradient(d_loss, discriminator_model.trainable_variables)
        
        ## Optimization: Apply the gradients
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, discriminator_model.trainable_variables))
                           
        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), 
             d_loss_real.numpy(), d_loss_fake.numpy()))
        
        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))
        epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))        
    all_losses.append(epoch_losses)
    all_d_vals.append(epoch_d_vals)
    print(
        'Epoch {:03d} | ET {:.2f} min | Avg Losses >>'
        ' G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake: {:.4f}]'
        .format(
            epoch, (time.time() - start_time)/60, 
            *list(np.mean(all_losses[-1], axis=0))))
    epoch_samples.append(
        create_samples(gen_model, fixed_z).numpy())