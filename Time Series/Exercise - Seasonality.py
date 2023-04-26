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
print(holidays_events.head())
