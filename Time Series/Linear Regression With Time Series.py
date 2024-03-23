import pandas as pd

df = pd.read_csv(
    "../input/ts-course-input/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
print(df.head())

import numpy as np

df['Time'] = np.arange(len(df.index))

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
fig.show()

df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])
print(df.head())
# target = weight * lag + bias
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(
    color='0.25'
))

ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales')
plt.show()
from pathlib import Path
from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter('ignore')
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True, figsize=(11, 4))
plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlepad=10,
)

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
#
data_dir = Path("../input/ts-course-input")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
print("0" * 100)

print(tunnel.head())
tunnel = tunnel.set_index('Day')
tunnel = tunnel.to_period()
print("1" * 100)
print(tunnel.head())
print("2" * 100)
df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))
print(df.head())

from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Time']]
y = df.loc[:, 'NumVehicles']

model = LinearRegression()
model.fit(X, y)
df['Time'] = np.arange(len(tunnel.index))

print(df.head())
from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Time']]
y = df.loc[:, 'NumVehicles']
#
# model = LinearRegression()
# model.fit(X, y)
# y_pred = pd.Series(model.predict(X), index=X.index)
print("X" * 100)
print(X.head())
print("y" * 100)
print(y.head())
y_pred = pd.Series(model.predict(X), index=X.index)
print("ypred," * 100)
print(y_pred.head())
df['Lag_1'] = df['NumVehicles'].shift(1)
print("df.head()," * 100)
print(df.head())
df['Lag_1'] = df['NumVehicles'].shift(1)
df.head()
from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'NumVehicles']
y, X = y.align(X, join='inner')
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
ax.plot(X['Lag_1'], y, '.', color='0.25')
X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:,'NumVehicles']
y, X = y.align(X, join='inner')
print("5"*100)
print(y.head())
print("6"*100)
print(X.head())

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');