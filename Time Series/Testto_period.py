import pandas as pd
import numpy as np

rng = pd.date_range('1/1/2000', periods=3, freq='M')
ts = pd.Series(
    np.random.randn(3),
    index=rng
)
print(ts)
pts = ts.to_period()
print("0" * 100)
print(pts)
print("1" * 100)
rng = pd.date_range('1/29/2000', periods=6, freq='D')
print(rng)
pd.Series(np.random)
# http://www.ay1.cc/article/1674035795786956816.html