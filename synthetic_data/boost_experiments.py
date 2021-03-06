import numpy as np
import pandas as pd
from process_data import process_data
from enhance_data import enhance_data
from nn_gridsearch import nn_gridsearch, make_model
from scipy.stats import reciprocal
from pprint import pprint


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
data_rate = 1.

# boost data progressively bigger to check results
for data_boost_x in [3, 5, 10, 20]:

    # progressive rate of the full training set
    for wasserstein in (True, False):

        # test each rate with boosted and non-boosted data
        for boost in (True,):

            print()
            print('================')
            print(f'creating data: {data_rate} of data boosted: {boost}')
            print(f'boosting real data times: {data_boost_x + 1}')

            data = enhance_data(synthetic_share=data_boost_x, include_synthetic=boost, wasserstein=wasserstein)

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
                'number_training_samples': data['y_train'].shape[0],
                'boostint_type': 'not_boosted' if not boost else 'wasserstein' if wasserstein else 'dcgan'}

            print('finished grid - results:')
            pprint(results)
            models_tested.append(results)

df = pd.DataFrame(models_tested)
#df = df.drop_duplicates(subset=['share_real_data', 'boosted_data', 'data_boosted_x', 'boostint_type']).sort_values(['share_real_data', 'data_boosted_x'])

with open('../notebooks/boosting_results_extreme.csv', mode='w') as file:
    df.to_csv(file)
