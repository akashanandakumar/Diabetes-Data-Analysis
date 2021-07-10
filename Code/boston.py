import numpy as np
import pandas as pd
bdata=pd.read_csv(r"C:\Users\Asus\Downloads\0000000000002417_training_boston_x_y_train (1).csv")
bdata

x_train_boston_data=bdata.iloc[:,0:13]
x_train_boston_data

y_train_boston_data=bdata.iloc[:,-1]
y_train_boston_data

from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
model.fit(x_train_boston_data,y_train_boston_data)
boston_data1=np.loadtxt(r"C:\Users\Asus\Downloads\0000000000002417_test_boston_x_test (2).csv",delimiter = ',')
y_predict=model.predict(boston_data1)
np.savetxt("boston_predict.csv", y_predict)

