# lists to store losses and values
all_losses = []
all_d_vals = []

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

if export_generator:
    
    print()
    print('saving generator model')
    
    tf.keras.models.save_model(generator_model, f'./models/generator_{model_name}.h5')
    print(f'generator model saved to: ./models/{model_name}.h5')
        
return result

