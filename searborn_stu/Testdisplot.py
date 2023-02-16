import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# sns.set(style="ticks", color_codes=True)
# tips = pd.read_csv("tips.csv")
sns.set_theme()
data = pd.read_csv('penguins.csv')
print(data.head())
print(data.columns)
# sns.displot(input=input['flipper_length_mm'])
# sns.histplot(input['flipper_length_mm'], kde=True, stat='density')
# sns.histplot(input['flipper_length_mm'], kde=True, stat='density',kde_kws=dict(cut=3))
# sns.histplot(
#     input['flipper_length_mm'],
#     kde_kws=dict(cut=3),
#     kde=True,
#     stat='density',
#     alpha=.4,
#     edgecolor=(1, 1, 1, .4),
# )
# sns.distplot(input['flipper_length_mm'], hist=False)
# sns.kdeplot(input["flipper_length_mm"])
# sns.displot(input["flipper_length_mm"], kind="kde")
# sns.histplot(input, x='flipper_length_mm', alpha=.4)
# sns.rugplot(input,x='flipper_length_mm')

sns.displot(data=data,x='flipper_length_mm',alpha=.4 ,rug=True)
plt.show()
