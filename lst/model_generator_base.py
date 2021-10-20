def make_model(
    input_shape: tuple = (11, ),
    number_hidden_layers: int = 8, 
    activation: str = 'elu', 
    alpha: float = .2,
    neurons: int = 32,
    loss: str = 'binary_crossentropy',
    learning_rate: float = .003,
    dropout_rate: float = .5) -> Model:

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for i in range(number_hidden_layers):
        model.add(layers.Dense(
            neurons, 
            kernel_initializer='he_normal',
            name=f'hidden_layer_{i}_relu_alpha_{alpha}'))
        
        if number_hidden_layers >= 3:
            model.add(layers.BatchNormalization())
        
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i}_{round(dropout_rate * 100)}'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', Precision(), Recall(), AUC()])
    
    return model
