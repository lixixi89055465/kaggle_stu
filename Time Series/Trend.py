from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
data_dir = Path('../input/ts-course-input')
tunnel = pd.read_csv(data_dir / 'tunnel.csv', parse_dates=['Day'])
print(tunnel.head())
tunnel = tunnel.set_index('Day').to_period()
print("0" * 100)
print(tunnel.head())
print("1" * 100)
print(tunnel.shape)
moving_average = tunnel.rolling(
    window=365,  # 365-day window
    center=True,  # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()  # compute the mean (could also do median, std, min, max, ...)
ax = tunnel.plot(style='.', color='0.5')
moving_average.plot(ax=ax, linewidth=3, title='Tunnel Traffic - 365-Day moving average ', legend=False)
plt.show()
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training input
    constant=True,
    order=1,
    drop=True,
)

X = dp.in_sample()
print("2" * 100)
print(X.head())
from sklearn.linear_model import LinearRegression

y = tunnel['NumVehicles']
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
print("3" * 100)
print(y_pred.shape)
print(y_pred.head())

ax = tunnel.plot(style='.', color='0.5', title='Tunnel Traffic - Linear Trend')
_ = y_pred.plot(ax=ax, linewidth=3, label='Trend')
plt.show()
print("4" * 100)
X = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(X), index=X.index)
print(y_fore.shape)
print(y_fore.head())
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
