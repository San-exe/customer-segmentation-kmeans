import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
features = ['GrLivArea','OverallQual','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
#here im including only few constraints of dataset
x = df[features]
y = df['SalePrice']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#for missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

#creating model and training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predictions
y_pred = regressor.predict(x_test)
comparison = pd.DataFrame({
    'Predicted':y_pred,
    'Actual':y_test.values
})

print(comparison)

#evaluation
from sklearn.metrics import r2_score,mean_absolute_error
print("R2_Score:",r2_score(y_test,y_pred))
print("Mean_Absolute_error:",mean_absolute_error(y_test,y_pred))

#plotting into graph
plt.figure(figsize=(8,8))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],color = "lime")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted House Prices")
plt.show()