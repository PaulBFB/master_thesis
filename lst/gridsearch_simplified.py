def nn_gridsearch(
    make_model_function,
    x_train: np.ndarray = None,
    y_train: np.ndarray = None,
    params: dict = None,
    epochs: int = 100,
    validation_split: float = .1,
    patience: int = 10,
    batch_size: int = 16,
    n_iterations: int = 10,
    early_stop: bool = True,
    save_logs: bool = False,
    verbose: int = 1):
    
    keras_cl = KerasClassifier(
        make_model_function,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose)
    
    rnd_search_cv = RandomizedSearchCV(
        keras_cl, 
        params, 
        n_iter=n_iterations, 
        cv=3, 
        verbose=2, 
        n_jobs=-1)
    
    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(patience=patience, monitor='val_loss', mode='min'))
    if save_logs:
        callbacks.append(TensorBoard(logdir()))
    
    rnd_search_cv.fit(
        x_train, y_train, 
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks)
    
    return rnd_search_cv


if __name__ == '__main__':

    grid_parameters = {
        'number_hidden_layers': list(range(1, 8)),
        'neurons': np.arange(1, 100).tolist(),
        'learning_rate': reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
        'dropout_rate': np.arange(.2, .6, .1).tolist(),
        'alpha': np.arange(.2, .35, .05).tolist(),
        'activation': ['elu', 'selu', 'relu']}

    grid = nn_gridsearch(
        make_model, 
        data['x_train_processed'], data['y_train'],
        grid_parameters,
        n_iterations=3)

    best_model = grid.best_estimator_.model
    best_model.save(model_path)
