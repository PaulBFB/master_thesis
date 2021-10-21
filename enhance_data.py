import numpy as np
import tensorflow as tf
from process_data import process_data
from generate_data import generate_data
from train_generator import train_generator
from train_wasserstein_generator import train_generator as train_wasserstein_generator


data = process_data()


def enhance_data(
    x_train: np.array = data['x_train_processed'],
    y_train: np.array = data['y_train'],
    include_synthetic: bool=True,
    force_generator: bool=False,
    generator_epochs: int=40,
    synthetic_share: float=0.2,
    wasserstein: bool=False,
    replace_real_data: bool=False,
    real_share: float=1.0) -> dict:
    
    assert real_share <= 1, 'can only take 100% of all real data'

    if real_share != 1.0:
        # shorten real data
        number_real_samples = int(x_train.shape[0] * real_share)
        # create permutation index in order to not just omit last samples in order
        real_permutated_index = np.random.permutation(x_train.shape[0])
        shortened_index = real_permutated_index[:number_real_samples]
    
        x_train = x_train.take(shortened_index, axis=0)
        y_train = y_train.take(shortened_index, axis=0)
        
        # important! if real data is made smaller, the GENERATOR NEEDS TO BE RETRAINED
        # otherwise, information will implicitly "leak" from the complete training set into the synthetic data    
    else:
        pass
    
    if include_synthetic:
        
        if real_share < 1 or force_generator:
            print()
            print(f'fitting new {"wasserstein" if wasserstein else ""} generator on smaller data')
            if wasserstein:
                generator = train_wasserstein_generator(
                    training_data=np.column_stack((x_train, y_train)),
                    generate_img=True,
                    export_generator=False,
                    n_epochs=generator_epochs)
                generator = generator['generator']
            else:
                generator = train_generator(
                    training_data=np.column_stack((x_train, y_train)),
                    generate_img=True,
                    export_generator=False,
                    n_epochs=generator_epochs)
                generator = generator['generator']

        else:
            print()
            print(f'loading pre-trained {"wasserstein" if wasserstein else "dcgan"} generator')
            generator = tf.keras.models.load_model(f'./models/best_{"wasserstein_" if wasserstein else ""}generator.h5', compile=False)
    
        number_samples = x_train.shape[0] * synthetic_share
        number_samples = int(number_samples)
        synthetic_data = generate_data(model=generator, 
                                       number_samples=number_samples)
        
        if replace_real_data:
            x_train = synthetic_data['x_train']
            y_train = synthetic_data['y_train']
            
        else:
            # stack data together
            x_train = np.vstack((x_train, synthetic_data['x_train']))
            y_train = np.concatenate((y_train, synthetic_data['y_train']))
            
            # create random index permutation to shuffle both arrays randomly, but preserve label match
            permutation_index = np.random.permutation(x_train.shape[0])
            
            # overwrite with shuffled arrays along the index
            x_train = x_train.take(permutation_index, axis=0)
            y_train = y_train.take(permutation_index, axis=0)
    
    else:
        pass
    
    result = {
        'x_train_processed': x_train, 
        'y_train': y_train}    
    
    return result


if __name__ == '__main__':
    data = enhance_data(
        include_synthetic=True,
        wasserstein=False, 
        force_generator=True,
        replace_real_data=True)

    for k, v in filter(lambda x: x[0] != 'pipeline', data.items()):
        
        np.save(f'./data/titanic_{k}.npy', v)
        print(f'{k} -  {type(v)} - {v.shape}')
