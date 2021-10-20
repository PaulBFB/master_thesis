for epoch in range(1, n_epochs+1):
    epoch_losses, epoch_d_vals = [], []
    for i,(input_z,input_real) in enumerate(training_data):
        
                    
        # set up tapes for both models
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            g_output = generator_model(input_z, training=True)
            
            # real and fake part of the critics output
            d_critics_real = discriminator_model(input_real, training=True)
            d_critics_fake = discriminator_model(g_output, training=True)
                
            # generator loss - (reverse of discriminator, to avoid vanishing gradient)
            g_loss = -tf.math.reduce_mean(d_critics_fake)
                
            # discriminator losses                    
            d_loss_real = -tf.math.reduce_mean(d_critics_real)
            d_loss_fake =  tf.math.reduce_mean(d_critics_fake)
            d_loss = d_loss_real + d_loss_fake
                
            # INNER LOOP for gradient penalty based on interpolations
            with tf.GradientTape() as gp_tape:
                alpha = rng.uniform(
                    shape=[d_critics_real.shape[0], 1, 1, 1], 
                    minval=0.0, maxval=1.0)
                
                # creating the interpolated examples
                interpolated = (
                    alpha*tf.cast(input_real, dtype=tf.float32) + (1-alpha)*g_output)
                
                # force recording of gradients of all interpolations (not created by model)
                gp_tape.watch(interpolated)
                d_critics_intp = discriminator_model(interpolated)
                
            # gradients of the discriminator w. regard to all
            grads_intp = gp_tape.gradient(
                d_critics_intp, [interpolated,])[0]
                
            # regularization
            grads_intp_l2 = tf.sqrt(
                tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
                
            # compute penalty w. lambda hyperparam
            grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))
                
            # add GP to discriminator
            d_loss = d_loss + lambda_gp*grad_penalty
