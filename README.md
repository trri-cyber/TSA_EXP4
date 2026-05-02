# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02-05-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv("/content/global_inflation_post_covid.csv")
data = data[data['country'] == 'USA']
data['date'] = pd.to_datetime(data['date'])
data = data.groupby('date')['food_price_index'].mean().reset_index()

X = data['food_price_index']

plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)

plt.title('Original inflation of Food price index postcovid')
plt.xlabel('Date')
plt.ylabel('Food index')

plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plot_acf(X, lags=29, ax=plt.gca())

plt.title('Original Data ACF')

plt.subplot(2, 1, 2)

plot_pacf(X, lags=29, ax=plt.gca())

plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Model Summary")
print(arma11_model.summary())

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))

plt.plot(ARMA_1)

plt.title('Simulated ARMA(1,1) inflation of Food price index postcovid')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

plot_acf(ARMA_1)

plt.title('ACF of ARMA(1,1)')

plt.show()

plot_pacf(ARMA_1)

plt.title('PACF of ARMA(1,1)')

plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Model Summary")
print(arma22_model.summary())

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 6))

plt.plot(ARMA_2)

plt.title('Simulated ARMA(2,2) inflation of Food price index postcovid')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

plot_acf(ARMA_2)

plt.title('ACF of ARMA(2,2)')

plt.show()

plot_pacf(ARMA_2)

plt.title('PACF of ARMA(2,2)')

plt.show()
```
OUTPUT:
ORIGINAL :

<img width="998" height="542" alt="image" src="https://github.com/user-attachments/assets/8c88afa3-f4a0-4e17-96e0-185e8a402f44" />

Partial Autocorrelation

<img width="1180" height="385" alt="image" src="https://github.com/user-attachments/assets/bd856528-799b-4c6e-8872-a14a9105a2d4" />

Autocorrelation

<img width="1182" height="395" alt="image" src="https://github.com/user-attachments/assets/784353cc-ade9-4b29-a675-7b9adf6deb35" />


SIMULATED ARMA(1,1) PROCESS:

<img width="1006" height="533" alt="image" src="https://github.com/user-attachments/assets/0fd259db-ba11-4f5b-b305-902a15d81d8e" />


Partial Autocorrelation

<img width="1001" height="515" alt="image" src="https://github.com/user-attachments/assets/89f20df4-9ec3-4d96-9eaa-52764d54f8b2" />


Autocorrelation

<img width="995" height="530" alt="image" src="https://github.com/user-attachments/assets/a871bfb4-9c5b-4bc4-99a0-107610a48a93" />


SIMULATED ARMA(2,2) PROCESS:

<img width="1005" height="536" alt="image" src="https://github.com/user-attachments/assets/cb32685e-67c9-450d-b2cd-a711f0d33421" />


Partial Autocorrelation

<img width="993" height="516" alt="image" src="https://github.com/user-attachments/assets/e5bac006-9d18-4962-9d21-4635dfea5986" />


Autocorrelation

<img width="998" height="537" alt="image" src="https://github.com/user-attachments/assets/a67fd426-7258-4d13-a16e-3b801be6345a" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
