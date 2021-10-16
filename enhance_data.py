import numpy as np
import tensorflow as tf
from process_data import process_data
from generate_data import generate_data
from train_generator import train_generator


def enhance_data(
    include_synthetic: bool=True,
    synthetic_share: float=0.2,
    real_share: float=1.0):
    
    assert real_share <= 1, 'can only take 100% of all real data'

    data = process_data()
    x_train_processed = data['x_train_processed']
    y_train = data['y_train']
    
    if real_share != 1.0 and include_synthetic:
        # shorten real data
        number_real_samples = int(x_train_processed.shape[0] * real_share)
        # create permutation index in order to not just omit last samples in order
        real_permutated_index = np.random.permutation(x_train_processed.shape[0])
        shortened_index = real_permutated_index[:number_real_samples]
    
        x_train_processed = x_train_processed.take(shortened_index, axis=0)
        y_train = y_train.take(shortened_index, axis=0)
        
        # important! if real data is made smaller, the GENERATOR NEEDS TO BE RETRAINED
        # otherwise, information will implicitly "leak" from the complete training set into the synthetic data
        print('fitting new generator on smaller data')
        generator = train_generator(
            training_data=np.column_stack((x_train_processed, y_train)),
            generate_img=False,
            export_generator=False)
        generator = generator['generator']
        
    else:
        generator = tf.keras.models.load_model('./models/best_generator.h5')
    
    if include_synthetic:
        number_samples = x_train_processed.shape[0] * synthetic_share
        number_samples = int(number_samples)
        synthetic_data = generate_data(model=generator, 
                                       number_samples=number_samples)
        
        # stack data together
        x_train_processed = np.vstack((x_train_processed, synthetic_data['x_train']))
        y_train = np.concatenate((y_train, synthetic_data['y_train']))
        
        # create random index permutation to shuffle both arrays randomly, but preserve label match
        permutation_index = np.random.permutation(x_train_processed.shape[0])
        
        # overwrite with shuffled arrays along the index
        x_train_processed = x_train_processed.take(permutation_index, axis=0)
        y_train = y_train.take(permutation_index, axis=0)
    
    else:
        pass
    
    result = {
        'x_train_processed': x_train_processed, 
        'y_train': y_train,
        'x_test': data['x_test'], 
        'x_test_processed': data['x_test_processed'], 
        'y_test': data['y_test']}    
    
    return result


if __name__ == '__main__':
    data = enhance_data(include_synthetic=False, real_share=0.5)

    for k, v in filter(lambda x: x[0] != 'pipeline', data.items()):
        
        np.save(f'./data/titanic_{k}.npy', v)
        print(f'{k} -  {type(v)} - {v.shape}')
