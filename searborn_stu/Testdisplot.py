import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# sns.set(style="ticks", color_codes=True)
# tips = pd.read_csv("tips.csv")
sns.set_theme()
data = pd.read_csv('penguins.csv')
print(data.head())
print(data.columns)
# sns.displot(data=data['flipper_length_mm'])
# sns.histplot(data['flipper_length_mm'], kde=True, stat='density')
# sns.histplot(data['flipper_length_mm'], kde=True, stat='density',kde_kws=dict(cut=3))
# sns.histplot(
#     data['flipper_length_mm'],
#     kde_kws=dict(cut=3),
#     kde=True,
#     stat='density',
#     alpha=.4,
#     edgecolor=(1, 1, 1, .4),
# )
# sns.distplot(data['flipper_length_mm'], hist=False)
# sns.kdeplot(data["flipper_length_mm"])
# sns.displot(data["flipper_length_mm"], kind="kde")
# sns.histplot(data, x='flipper_length_mm', alpha=.4)
# sns.rugplot(data,x='flipper_length_mm')

sns.displot(data=data,x='flipper_length_mm',alpha=.4 ,rug=True)
plt.show()
