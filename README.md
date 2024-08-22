# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 22/08/24
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:

```py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
```
```py
data = pd.read_csv('/content/waterquality.csv')
```

```py
data['Date'] = pd.to_datetime(data['Date'])
```
```py
data['pH'] = data['pH'].fillna(data['pH'].mean())
```
```py
X = np.array(data.index).reshape(-1, 1)
y = data['pH']
```

```py
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)
```

```py
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

```py
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)
```

```py
plt.figure(figsize=(35, 5))
plt.subplot(1,3,1)
plt.plot(data['Date'], data['pH'], label='pH level')
plt.xlabel('Date')
plt.ylabel('pH level')
plt.title('Year-wise pH level Over Time')
plt.grid(True)

plt.figure(figsize=(35, 5))
plt.subplot(1,3,2)
plt.plot(data['Date'], y, label='pH level')
plt.plot(data['Date'], y_pred_linear, color='red',linestyle='--', label='Linear Trend')
plt.xlabel('Date')
plt.ylabel('pH level')
plt.title('Linear Trend Estimation for pH level')
plt.legend()
plt.grid(True)


plt.figure(figsize=(35, 5))
plt.subplot(1,3,3)
plt.plot(data['Date'], y, label='Actual pH level')
plt.plot(data['Date'], y_pred_poly, color='green',linestyle='--', label='Polynomial Trend (Degree 2)')
plt.xlabel('Date')
plt.ylabel('pH level')
plt.title('Polynomial Trend Estimation for pH level')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT

![image](https://github.com/user-attachments/assets/46ffec37-a1ac-49cf-b2d7-bb931bb1a29a)


A - LINEAR TREND ESTIMATION

![image](https://github.com/user-attachments/assets/f9b399f1-7a0d-4375-a33e-4934eceb0e24)



B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/e40a173e-e98c-4d37-98d0-dc831ec6150b)



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
