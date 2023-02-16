import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", color_codes=True)
tips = pd.read_csv("tips.csv")
print(tips.head())

# distplot
# sns.displot(tips['total_bill'],hist_kws={'cumulative':False}, kde_kws={'cumulative':False}    )
# sns.distplot(tips['total_bill'],hist_kws={'cumulative':True},
#              kde_kws={'cumulative':True},
#              )
print(help(sns.displot))
# sns.displot(input=tips,x='total_bill',y='day')
sns.displot(data=tips,y='total_bill',x='day')
plt.show()
# displot
