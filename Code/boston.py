import numpy as np
import pandas as pd
bdata=pd.read_csv(r"FOLDER_NAME.csv")
bdata

x_train_boston_data=bdata.iloc[:,0:13]
x_train_boston_data

y_train_boston_data=bdata.iloc[:,-1]
y_train_boston_data

from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
model.fit(x_train_boston_data,y_train_boston_data)
boston_data1=np.loadtxt(r"FOLDER_NAMEcsv",delimiter = ',')
y_predict=model.predict(boston_data1)
np.savetxt("boston_predict.csv", y_predict)

