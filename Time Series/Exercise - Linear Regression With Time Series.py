from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

data_dir = Path('../input/ts-course-input/')
comp_dir = Path('../input/store-sales-time-series-forecasting')
book_sales = pd.read_csv(
    data_dir / 'book_sales.csv',
    index_col='Date',
    parse_dates=['Date']
).drop('Paperback', axis=1)

print(book_sales.head())
book_sales['Time'] = np.arange(len(book_sales.index))
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])
print(book_sales.head())
ar = pd.read_csv(data_dir / 'ar.csv')
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
print("0" * 100)
print(store_sales.head())
print("1" * 100)
store_sales = store_sales.set_index('date').to_period('D')
print(store_sales.head())
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
print("2" * 100)
print(store_sales.head())
print("3" * 100)
average_sales = store_sales.groupby('date').mean()['sales']
print(average_sales.head())
# fig, ax = plt.subplots()
# ax.plot('Time', 'HardCover', input=book_sales, color='0.75')
# ax.plot('Time', 'Hardcover', input=book_sales, color='0.75')
# ax = sns.regplot(x='Time', y='Hardcover', input=book_sales, ci=None,
#                  scatter_kws=dict(color='0.25'))
# ax.set_title('Time plot of hardcover sales')
# plt.show()
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
# ax1.plot(ar['ar1'])
# ax1.set_title('Series 1')
# ax2.plot(ar['ar2'])
# ax2.set_title('Series 2')
print("4" * 100)
from sklearn.linear_model import LinearRegression

df = average_sales.to_frame()
print(df.head())
time = np.arange(len(df.index))
df['time'] = time
X = df.loc[:, ['time']]
y = df.loc[:, 'sales']
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
# ax = y.plot(**plot_params, alpha=0.5)
# ax = y_pred.plot(ax=ax, linewidth=3)
# ax.set_title('Time Plot of Total Store Sales');
# plt.show()
print("6" * 100)
df = average_sales.to_frame()
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1
X = df.loc[:, ['lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'sales']

y, X = y.align(X, join='inner')
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
