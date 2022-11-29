import pandas as pd
import numpy as np

s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
print(s)
print(s.isin(['cow', 'lama']))
print(s.isin(['lama']))
