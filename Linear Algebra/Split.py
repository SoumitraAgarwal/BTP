import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
target 	= data[['Y']]
data 	= data[['X1', 'X2']]
data 	= data.values
target 	= target.values


X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

data	= np.hstack([X_train, y_train])
data 	= pd.DataFrame(data, columns = ['X1', 'X2', 'Y'])
data.to_csv('data.csv', index = False)

data	= np.hstack([X_test, y_test])
data 	= pd.DataFrame(data, columns = ['X1', 'X2', 'Y'])
data.to_csv('predict.csv', index = False)