import math
from tensorflow import keras
from nn_gridsearch import logdir


K = keras.backend


class OneCycleScheduler(keras.callbacks.Callback):
    """
    a keras callback learning rate scheduler that implements the 1cycle learning rate proposed by Leslie N. Smith 
    in her 2018 paper 'a disciplined approach to neural network hyper-parameters' https://arxiv.org/abs/1803.09820
    
    inherits from keras callback to adjust momentum on batch start
    """

    def __init__(
        self, 
        iterations: int, 
        max_rate: float, 
        start_rate: float=None,
        last_iterations: int=None, 
        last_rate: float=None):
        
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    
    def on_batch_begin(self, batch, logs):
        
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)
        
        
if __name__ == '__main__':
    from process_data import process_data
    
    data = process_data()
    
    x_train = data['x_train_processed']
    y_train = data['y_train']
    
    epochs = 100
    batch_size = 16
    
    model = keras.models.load_model('./models/titanic_nn.h5')
    
    one_cycle = OneCycleScheduler(
        iterations=math.ceil(len(x_train) / batch_size) * epochs, 
        max_rate=.05)
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=.2, 
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.TensorBoard(logdir('one_cycle_lr_schedule')), 
            one_cycle])
