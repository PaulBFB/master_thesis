import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def process_data(
    include_synthetic: bool=True):
    
    with open('./data/titanic.csv', mode='r') as file:
        df = pd.read_csv(file)

    df.drop(columns=['cabin', 'home.dest', 'boat', 'body', 'ticket'], inplace=True)
    df['title'] = df['name'].str.extract('([A-Za-z]+)\.', expand=True)
    
    df['high_status'] = np.where(df['title'].isin([
        'Dr', 
        'Rev', 
        'Col', 
        'Major', 
        'Lady', 
        'Countess', 
        'Dona', 
        'Don', 
        'Capt', 
        'Sir' 
        'Jonkheer']),
                                 1, 0)
    
    df['married'] = np.where(df['title'].isin([
        'Mrs', 
        'Lady', 
        'Countess', 
        'Dona', 
        'Don',]),
                             1, 0)
    
    df.drop(columns=['title', 'name'], inplace=True)
    df.drop(labels=1309, inplace=True)
    
    y = df.pop('survived')
    x = df
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.2)
    
    
    transformer = ColumnTransformer([
    ('onehotencode categories', OneHotEncoder(), ['sex', 'embarked']),
    ('normally distributed', StandardScaler(), ['age']),
    ('nor normally distributed', MinMaxScaler(), ['pclass', 'sibsp', 'parch', 'fare'])], 
    remainder='drop')
    
    # chain transformer and imputation
    pipeline = Pipeline([('transform', transformer), ('impute', IterativeImputer(min_value=0))])
    x_train_processed = pipeline.fit_transform(x_train)
    x_test_processed = pipeline.fit_transform(x_test)
    
    result = {
        'x_train': x_train.values, 
        'x_train_processed': x_train_processed, 
        'x_test': x_test.values, 
        'x_test_processed': x_test_processed, 
        'y_train': y_train.values,
        'y_test': y_test.values,
        'pipeline': pipeline}    
    
    return result


if __name__ == '__main__':
    data = process_data()

    for k, v in filter(lambda x: x[0] != 'pipeline', data.items()):
        
        np.save(f'./data/titanic_{k}.npy', v)
        print(k)
    