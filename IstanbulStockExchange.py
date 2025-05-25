import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

data = pd.read_csv('ISE.csv')

# Define your features (X) and target variable (y)
X = data[['SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM']]
y = data['ISE']
print(y)
print(X)
plt.figure(figsize = (7,7))
plt.scatter (X['SP'], y)
plt.title("S&P 500 vs Istanbul Stock Exchange")
plt.xlabel("S&P 500")
plt.ylabel("ISE")
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R-squared:', r2)

# Get the coefficients and features
coefficients = model.coef_
intercept = model.intercept_
features = model.feature_names_in_

# Find the index of the highest coefficient
max_index = np.argmax(coefficients)

# Get the highest coefficient and corresponding feature
max_coefficient = coefficients[max_index]
max_feature = features[max_index]

print("Feature with the highest coefficient:", features[max_index],"=",max_coefficient)

plt.title('Actual Data vs. Predictions')
plt.scatter (y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predictions')
#plt.show()

# Step 7: use the model
new_data = pd.DataFrame(([[0.01,0.01,0.01,0.01,0.01,0.01,0.01]]), columns = ['SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM'])
print(new_data)

new_pred = model.predict(new_data)
print(new_pred)

print(coefficients)
print(intercept)



