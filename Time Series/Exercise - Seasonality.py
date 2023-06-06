# Setup feedback system
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

comp_dir = Path("../input/store-sales-time-series-forecasting")
holidays_events = pd.read_csv(
    comp_dir / 'holidays_events.csv',
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
print(holidays_events.columns)
print(holidays_events.dtypes)
print('0' * 100)
holidays_events.set_index('date').to_period('D')
print(holidays_events.head())
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True
)

print('1' * 100)
print(store_sales.head())
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales.groupby('date').mean().squeeze().loc['2017']
)

X = average_sales.to_frame()
X['week'] = X.index.week
X['day'] = X.index.dayofweek
# seasonal_plot(X, y='sales', period='week', freq='day')

