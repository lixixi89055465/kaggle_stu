from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data_dir = Path('../input/ts-course-data/')
comp_dir = Path('../input/store-sales-time-series-forecasting')
retail_sales = pd.read_csv(
    data_dir / 'us-retail-sales.csv',
    parse_dates=['Month'],
    index_col='Month'
).to_period('D')
print(retail_sales.head())
print(retail_sales.columns)
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']
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
    infer_datetime_format=True
)
print("2" * 100)
print(store_sales.head())
store_sales = store_sales.set_index('date').to_period('D')
print(store_sales.head())
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
print("3" * 100)
print(store_sales.head())
average_sales = store_sales.groupby('date').mean()['sales']
print("4" * 100)
print(average_sales.head())
print("5"*100)
ax = food_sales.plot(**plot_params)
ax.set(title="US Food and Beverage Sales", ylabel="Millions of Dollars");
plt.show()