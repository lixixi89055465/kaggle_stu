import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", color_codes=True)
df= pd.read_csv("tips.csv")
print(df.head())
## 注意：连续变量作图的话，如果是x轴为连续变量，必须要用Building structed multi-plot grids
print(df.day.value_counts().index)
