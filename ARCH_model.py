#ARCH Model (Auto Regressive Conditional Heteroskedasticity Model)
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf

# Load data from CSV file
data = pd.read_csv('C:/Users/SHAIFALI PATWAL/Desktop/Github Projects/stock_price_data.csv', parse_dates=['Date'])

# Plotting the stock Price data
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Price'], label='Original Data')
plt.title('Stock Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# we can observe the sudden burst of the volitility in the data
 #%%
# Splitting the data into train and test sets
train_data = data.iloc[:200]
train_data.head()
test_data = data.iloc[200:]
test_data.head()

 #%%
# Plotting PACF
plot_pacf(data['Price'], lags=20)

#Here in the PACF, at the threshold 0.10, we can observe that the lag 1 is significant therefore we’ll create an ARCH(1) model where p=1.
 #%%
# Fitting the ARCH model
model = arch_model(train_data['Price'], vol='ARCH', p=1)
model_fitted = model.fit()

# Forecast
forecast = model_fitted.forecast(horizon=3)
forecast
# Print summary of the model
print(model_fitted.summary())

# We can see both coeff. are significant as p-values are very low
 #%%
# Plot the forecast
model_fitted.plot()
plt.title('ARCH Model Forecast')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show() 


