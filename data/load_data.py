import pandas as pd
from scipy.io.arff import loadarff


data = loadarff('./messidor_features.arff')
df = pd.DataFrame(data[0])
print(df.info())
