import time
import itertools
import tensorflow as tf
import numpy as np
from process_data import process_data
from matplotlib import pyplot as plt


# https://www.tensorflow.org/guide/random_numbers
tf.random.set_seed(42)
np.random.seed(42)

if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'

    
def create_generator_network(
    number_hidden_layers: int = 1,
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
            # implements the guideline from hands on ML - upsampling layers in the generator
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


@tf.function
def preprocess(
    passenger: np.ndarray,
    rng: tf.random.Generator,
    output_shape: tuple = (12,),
    latent_space_shape: tuple = (12,),
    mode: str='uniform'):
    
    assert passenger.shape == output_shape, f'data shape needs to match the expected output shape {output_shape} - shape is {passenger.shape}'
    
    passenger = tf.convert_to_tensor(passenger)
    
    if mode == 'uniform':
        input_z = rng.uniform(
            shape=latent_space_shape, 
            minval=-1.0, maxval=1.0)
    
    elif mode == 'normal':
        input_z = rng.normal(shape=latent_space_shape)
    
    return input_z, passenger


def train_generator(
    training_data: np.ndarray,
    latent_space_shape: int=8,
    latent_space_mode: str='normal',
    rng: tf.random.Generator=tf.random.Generator.from_seed(42),
    number_hidden_layers: int = 2,
    number_hidden_units_power: int = 5,
    hidden_activation: str = 'selu',
    n_epochs: int = 40,
    batch_size: int = 32,
    tensorflow_device: str = device_name,
    generate_img: bool = True,
    learning_rate: float = 0.0005,
    export_generator: bool = True) -> tf.keras.Model:
    
    
    data_shape = training_data.shape[1]
    data_shape = (data_shape,)
    
    training_data = tf.data.Dataset.from_tensor_slices(training_data)
    training_data = training_data.map(lambda x: preprocess(x, rng=rng, latent_space_shape=(latent_space_shape,), mode=latent_space_mode, output_shape=data_shape))
            
    training_data = training_data.shuffle(10000)
    training_data = training_data.batch(
        batch_size, drop_remainder=True)
    
    with tf.device(tensorflow_device):
        generator_model = create_generator_network(
            number_hidden_layers=number_hidden_layers,
            number_hidden_units_power=number_hidden_units_power,
            hidden_activation_function=hidden_activation,
            number_output_units=np.product(data_shape))
        
        generator_model.build(input_shape=(None, latent_space_shape))
#        print(generator_model.summary())
        
        discriminator_model = create_discriminator_network(
            number_hidden_layers=number_hidden_layers,
            number_hidden_units_power=number_hidden_units_power,
            hidden_activation_function=hidden_activation)
        
        discriminator_model.build(input_shape=(None, np.prod(data_shape)))
#        print(discriminator_model.summary())
        
    # Loss functions and optimizers
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    # lists to store losses and values
    all_losses = []
    all_d_vals = []
    
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        epoch_losses, epoch_d_vals = [], []
        for i,(input_z,input_real) in enumerate(training_data):
            
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
    result = {
        'all_losses': all_losses,
        'all_d_vals': all_d_vals,
        'generator': generator_model,
        'discriminator': discriminator_model}

    model_name = f'e_{n_epochs}_layers_{number_hidden_layers}_units_{number_hidden_units_power}'

    
    if generate_img:
        
        print()
        print('generating training log image')
        
        fig = plt.figure(figsize=(20, 10))
    
        ## Plotting the losses
        ax = fig.add_subplot(1, 2, 1)
        g_losses = [item[0] for item in itertools.chain(*all_losses)]
        d_losses = [item[1]/2.0 for item in itertools.chain(*all_losses)]
        plt.plot(g_losses, label='Generator loss', alpha=0.75)
        plt.plot(d_losses, label='Discriminator loss', alpha=0.75)
        plt.legend(fontsize=20)
        ax.set_xlabel('Iteration', size=15)
        ax.set_ylabel('Loss', size=15)
        
        epochs = np.arange(1, n_epochs + 1)
        epoch2iter = lambda e: e*len(all_losses[-1])
        epoch_ticks = np.arange(0, n_epochs, 20)
        
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
        
        
        plt.savefig(f'../img/{model_name}.png')
        
        print(f'image saved to: ../img/{model_name}.png')
    
    if export_generator:
        
        print()
        print('saving generator model')
        
        tf.keras.models.save_model(generator_model, f'../models/generator_{model_name}.h5')
        print(f'generator model saved to: ../models/{model_name}.h5')
        
    return result


if __name__ == '__main__':
    
    data = process_data()    
    x_train = data['x_train_processed']
    y_train = data['y_train']

    x_train = np.column_stack((x_train, y_train))

    
    result = train_generator(training_data=x_train)
