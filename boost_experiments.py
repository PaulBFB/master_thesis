import numpy as np
import pandas as pd
from process_data import process_data
from enhance_data import enhance_data
from nn_gridsearch import nn_gridsearch, make_model
from scipy.stats import reciprocal
from pprint import pprint


#include_synthetic = False
#synthetic_share = 3.0
#real_share = 0.2
    
#data = enhance_data(
#    include_synthetic=include_synthetic,
#    synthetic_share=synthetic_share,
#    real_share=real_share)
    
testing = process_data()
x_test = testing['x_test_processed']
y_test = testing['y_test']
    
grid_parameters = {
    'number_hidden_layers': list(range(1, 8)),
    'neurons': np.arange(1, 100).tolist(),
    'learning_rate': reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    'dropout_rate': np.arange(.2, .6, .1).tolist(),
    'alpha': np.arange(.2, .35, .05).tolist(),
    'activation': ['elu', 'selu', 'relu']}

models_tested = []

# boost data progressively bigger to check results
for data_boost_x in [.2, .3, .5, 1]:

    # progressive rate of the full training set
    for data_rate in np.linspace(.3, .8, 6):
        
        # test each rate with boosted and non-boosted data
        for boost in (True, False):            
                
            print()
            print('================')
            print(f'creating data: {data_rate} of data boosted: {boost}')
            print(f'boosting real data times: {data_boost_x + 1}')
    
            data = enhance_data(real_share=data_rate, synthetic_share=data_boost_x, include_synthetic=boost)
            
            print(f'real')
            print(f'created training data: {data["x_train_processed"].shape[0]} samples total')
            grid = nn_gridsearch(
                make_model, 
                data['x_train_processed'], data['y_train'],
                grid_parameters,
                n_iterations=20,
                verbose=0)
            
            best_model = grid.best_estimator_.model
            stats = best_model.evaluate(x_test, y_test)
            
            results = {
                'model': best_model,
                'data_boosted_x': data_boost_x + 1 if boost else 0,
                'accuracy': stats[1],
                'share_real_data': data_rate,
                'boosted_data': boost,
                'number_training_samples': data['y_train'].shape[0]}
            
            print('finished grid - results:')
            pprint(results)
            models_tested.append(results)
    
# note = the wrapper takes a FUNCTION as input!
#grid = nn_gridsearch(
#    make_model, 
#    data['x_train_processed'], data['y_train'],
#    grid_parameters,
#    n_iterations=33,
#    verbose=0)
    
#best_model = grid.best_estimator_.model
#best_model.evaluate(data['x_test_processed'], data['y_test'])

df = pd.DataFrame(models_tested)
df = df.drop_duplicates(subset=['share_real_data', 'boosted_data', 'data_boosted_x']).sort_values(['share_real_data', 'data_boosted_x'])

with open('./boosting_results_new.csv', mode='w') as file:
    df.to_csv(file)

