import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
#from scikit-learn.linear_model import LinearRegression 
import pandas as pd

dataframe1 = pd.read_csv('D:\Relocated Folders from C\Downloads\Salary_Data.csv')
print(dataframe1)

x = dataframe1[['YearsExperience']]
y = dataframe1[['Salary']]

Lin = LinearRegression()
Lin.fit(x,y)
y_pred = Lin.predict(x)

from sklearn.metrics import root_mean_squared_error, r2_score
print(root_mean_squared_error(y,y_pred))
print(r2_score(y,y_pred))
print(y_pred)
#plt.scatter(x, y)
#plt.show()