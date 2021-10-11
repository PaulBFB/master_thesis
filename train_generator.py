import time
import itertools
import tensorflow as tf
import numpy as np
from process_data import process_data
from matplotlib import pyplot as plt


tf.random.set_seed(42)
np.random.seed(42)

if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'

    
def create_generator_network(
    number_hidden_layers: int = 1,
    number_hidden_units: int = 100,
    hidden_activation_function: str = 'LeakyReLU',
    use_dropout: bool = False,
    dropout_rate: float = 0.3,
    number_output_units: int = 12,
    output_activation_function: str = 'tanh') -> tf.keras.Model:

    model = tf.keras.Sequential()
    for i in range(number_hidden_layers):
        model.add(tf.keras.layers.Dense(number_hidden_units, use_bias=False))
        model.add(tf.keras.layers.Activation(hidden_activation_function))
        
        if use_dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            pass

    model.add(tf.keras.layers.Dense(number_output_units))
    model.add(tf.keras.layers.Activation(output_activation_function))

    return model


def create_discriminator_network(
    number_hidden_layers: int = 1,
    number_hidden_units: int = 100,
    hidden_activation_function: str = 'LeakyReLU',
    use_dropout: bool = True,
    dropout_rate: float = 0.5,
    number_output_units: int = 1) -> tf.keras.Model:

    model = tf.keras.Sequential()

    for i in range(number_hidden_layers):
        model.add(tf.keras.layers.Dense(number_hidden_units,))
        model.add(tf.keras.layers.Activation(hidden_activation_function))
        
        if use_dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            pass

    model.add(tf.keras.layers.Dense(number_output_units, activation=None))

    return model


@tf.function
def preprocess(
    passenger: np.ndarray,
    output_shape: tuple = (12,),
    latent_space_shape: tuple = (12,),
    mode: str='uniform'):
    
    assert passenger.shape == output_shape, f'data shape needs to match the expected output shape {output_shape} - shape is {passenger.shape}'
    
    passenger = tf.convert_to_tensor(passenger)
    
    if mode == 'uniform':
        input_z = tf.random.uniform(
            shape=latent_space_shape, 
            minval=-1.0, maxval=1.0)
    
    elif mode == 'normal':
        input_z = tf.random.normal(shape=latent_space_shape)
    
    return input_z, passenger


# pull data from processing function
data = process_data()    
x_train = data['x_train_processed']
y_train = data['y_train']

# stack labels back to training data
x_train = np.column_stack((x_train, y_train))

# convert to tensorflow dataset, in order to map preprocess function on it
x_train_tf = tf.data.Dataset.from_tensor_slices(x_train)
x_train_tf = x_train_tf.map(preprocess)

# shape of output data - experiment with the relation, latent space should be smaller in order to compress data?
data_shape = (12,)
# dimension of the latent space
z_size = 12 
mode_z = 'normal'

# training params - :todo wrap into a function
num_epochs = 50
batch_size = 32

generator_hidden_layers = 1
generator_hidden_units = 100

discriminator_hidden_layers = 1
discriminator_hidden_units = 100

# setup latent space distribution
if mode_z == 'uniform':
    fixed_z = tf.random.uniform(
        shape=(batch_size, z_size),
        minval=-1, maxval=1)
elif mode_z == 'normal':
    fixed_z = tf.random.normal(
        shape=(batch_size, z_size))


def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *data_shape))    
    return (images+1)/2.0


x_train_tf = x_train_tf.shuffle(10000)
x_train_tf = x_train_tf.batch(
    batch_size, drop_remainder=True)

# create the models
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

# Loss functions and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

# lists to store losses and values
all_losses = []
all_d_vals = []
epoch_samples = []

start_time = time.time()
for epoch in range(1, num_epochs+1):
    epoch_losses, epoch_d_vals = [], []
    for i,(input_z,input_real) in enumerate(x_train_tf):
        
        # generator loss, record gradients
        with tf.GradientTape() as g_tape:
            g_output = generator_model(input_z)
            d_logits_fake = discriminator_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)
        
        # get loss derivatives from tabe, only for trainable vars, in case of regularization / batchnorm
        g_grads = g_tape.gradient(g_loss, generator_model.trainable_variables)
        
        # apply optimizer for generator
        g_optimizer.apply_gradients(
            grads_and_vars=zip(g_grads, generator_model.trainable_variables))

        # discriminator loss, gradients
        with tf.GradientTape() as d_tape:
            d_logits_real = discriminator_model(input_real, training=True)

            d_labels_real = tf.ones_like(d_logits_real)
            
            # loss for the real examples - labeles as 1
            d_loss_real = loss_fn(
                y_true=d_labels_real, y_pred=d_logits_real)

            # loss for the fakes - labeled as 0 
            
            # apply discriminator to generator output like a function
            d_logits_fake = discriminator_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)

            # loss function
            d_loss_fake = loss_fn(
                y_true=d_labels_fake, y_pred=d_logits_fake)

            # compute component loss for real & fake
            d_loss = d_loss_real + d_loss_fake

        # get the loss derivatives from the tape
        d_grads = d_tape.gradient(d_loss, discriminator_model.trainable_variables)
        
        # apply optimizer to discriminator gradients - only trainable :todo: add regularization here
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, discriminator_model.trainable_variables))
                           
        # add step loss to epoch list
        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), 
             d_loss_real.numpy(), d_loss_fake.numpy()))
        
        # probabilities from logits for predcitions, using tf builtin
        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))
        epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))        
    
    # record loss
    all_losses.append(epoch_losses)
    all_d_vals.append(epoch_d_vals)
    print(
        'Epoch {:03d} | ET {:.2f} min | Avg Losses >>'
        ' G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake: {:.4f}]'
        .format(
            epoch, (time.time() - start_time)/60, 
            *list(np.mean(all_losses[-1], axis=0))))
    epoch_samples.append(
        create_samples(generator_model, fixed_z).numpy())


samples = create_samples(generator_model, fixed_z).numpy()
np.save(f'./data/titanic_generated.npy', samples)

tf.keras.models.save_model(generator_model, './models/generator_latest.h5')

fig = plt.figure(figsize=(20, 10))

## Plotting the losses
ax = fig.add_subplot(1, 2, 1)
g_losses = [item[0] for item in itertools.chain(*all_losses)]
d_losses = [item[1]/2.0 for item in itertools.chain(*all_losses)]
plt.plot(g_losses, label='Generator loss', alpha=0.95)
plt.plot(d_losses, label='Discriminator loss', alpha=0.95)
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)

epochs = np.arange(1, num_epochs + 1)
epoch2iter = lambda e: e*len(all_losses[-1])
epoch_ticks = np.arange(0, num_epochs, 20)

newpos = [epoch2iter(e) for e in epoch_ticks]
ax2 = ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 60))
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)

# Plotting the outputs of the discriminator
ax = fig.add_subplot(1, 2, 2)
d_vals_real = [item[0] for item in itertools.chain(*all_d_vals)]
d_vals_fake = [item[1] for item in itertools.chain(*all_d_vals)]
plt.plot(d_vals_real, alpha=0.75, label=r'Real: $D(\mathbf{x})$')
plt.plot(d_vals_fake, alpha=0.75, label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator output', size=15)

ax2 = ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 60))
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)

plt.savefig('./img/gan_convergence.png')
