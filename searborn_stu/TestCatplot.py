import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", color_codes=True)
tips = pd.read_csv("tips.csv")
print(tips.head())
# catplot（） 内置Kind: stripplot()
# sns.catplot(x='day',y='total_bill',data=tips)

# 参数kind='box'

# sns.catplot(x='day',y='total_bill',kind='box',data=tips)
# sns.catplot(x='day',y='total_bill',kind='box',hue='smoker',data=tips)

# print("2" * 100)
# diamons = pd.read_csv('diamonds.csv')
# print(diamons.head())
# print(diamons.size)

# sns.catplot(x='color', y='price', kind='boxen',
#             data=diamons.sort_values('color') )
# 参数 kind="boxen" 适合大数据

# sns.catplot(x='total_bill', y='day', hue='time',
#             kind='violin', data=tips
#             )
# 参数 kind=“violin” ：它将箱线图和核密度估计(kde: kernel density estimation)结合起来

# print("3" * 100)
# titanic = pd.read_csv('titanic.csv')
# print(titanic.columns)
# print(titanic.head())
# sns.catplot(x='sex', y='survived', hue='class', kind='bar', data=titanic)
# print("4" * 100)
# sns.catplot(x='deck',
#             kind='count',
#             palette='ch:.25',
#             data=titanic)

print("5" * 100)
# sns.catplot(x='day',y='total_bill',data=tips,kind='strip')
# sns.stripplot(x='day',y='total_bill',data=tips)
print("6" * 100)
# sns.catplot(x='day',y='total_bill',data=tips,kind='point')
# sns.pointplot(x='day', y='total_bill', data=tips)
# 1、aggegation成平均（点图）
print("7" * 100)
# sns.catplot(x='day', y='conversion', data=tips, kind='bar', ci=None)
# sns.barplot(x='day', y='conversion', data=tips, ci=None)
print("8" * 100)
# 3、另外，catplot还能做max，min的distribtuion
# sns.catplot(x='day',y='total_bill',data=tips,kind='box')
# sns.boxplot(x='day',y='total_bill',data=tips)
print("9" * 100)
# sns.catplot(x='day',y='total_bill',data=tips,kind='violin')
# sns.violinplot(x='day',y='total_bill',data=tips)
print("9" * 100)
# 4、当然，像sns.catplot开头的那张也可以对y值直接做plot，不做任何的aggregation
# sns.catplot(x='day', y='total_bill', data=tips, kind='violin')
# sns.violinplot(x='day', y='total_bill', data=tips)
# 5、sns.plot还可直接计算categorical变量出现的次数：
print("a" * 100)
# sns.catplot(x='day',data=tips,kind='count')
sns.countplot(x='day',data=tips)

plt.show()
