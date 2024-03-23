# -*- coding: utf-8 -*-
# @Time : 2024/3/16 8:49
# @Author : nanji
# @Site : 
# @File : testSelectKBest.py
# @Software: PyCharm
# @Comment :https://zhuanlan.zhihu.com/p/141010878


import pandas as pd
# load sklearn built-in Boston dataset
from sklearn.datasets import load_boston
# Loading the dataset

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
columns = [
	'CRIM',
	'ZN',
	'INDUS',
	'CHAS',
	'NOX',
	'RM',
	'AGE',
	'DIS',
	'RAD',
	'TAX',
	'PTRATIO',
	'B',
	'LSTAT',
	# 'MEDV',
]
data = pd.DataFrame(data, columns=columns)
X = data
target = raw_df.values[1::2, 2]
# x = load_boston()
# input = pd.DataFrame(x.input, columns = x.feature_names)
# input["MEDV"] = target
# X = input.drop("MEDV", 1)  # Remove Target Variable to Get Feature Matrix
y = target
print(data.head())
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

bestfeaturs = SelectKBest(score_func=f_regression, k=5)
fit = bestfeaturs.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(5, 'Score'))

import seaborn as sns
from matplotlib import pyplot as plt

# get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
# plt.figure(figsize=(20, 20))
# plot heat map
# g=sns.heatmap(input[top_corr_features].corr(),annot=True,cmap="RdYlGn")


from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor()
# model.fit(X,y)
# print(model.feature_importances_)
# feat_importances=pd.Series(model.feature_importances_,index=X.columns)
# feat_importances.nlargest(5).plot(kind='barh')

from sklearn.feature_selection import RFE

# model = RandomForestRegressor()
# # rfe = RFE(model, 5)
# rfe = RFE(model)
# fit = rfe.fit(X, y)
# print('Num features : %s' % (fit.n_features_))
# print('Selecteed Features  : %s' % (fit.support_))
# print('Feature Ranking :%s' % (fit.ranking_))


from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
# rfecv=RFECV(RandomForestRegressor())
# rfecv.fit(X,y)
# print('Optimal number of features : %d' %rfecv.n_features_)
# plt.figure()
# plt.xlabel('Number of features selected')
# plt.ylabel('Cross validation score (nb of correct classification ')
# plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
# plt.show()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
#Build RF regressor to use in feature selection
# clf=RandomForestRegressor()
#Sequential Forward Selection
# sfs=sfs(
# 	clf,
# 	k_features=5,
# 	forward=True,\
# 	floating=False,\
# 	verbose=2,\
# 	scoring='neg_mean_squared_error',\
# 	cv=5 )
# sfs=sfs.fit(X,y)
# print('\n Sequential Forward selection (k=5):')
# print(sfs.k_feature_idx_)
# print('CV Score : ')
# print(sfs.k_score_)
# fig=plot_sfs(sfs.get_metric_dict(),kind='std_err')
# plt.title('Sequential Forward selection (w.StdError)')
# plt.grid()
# plt.show()
from sklearn.inspection import permutation_importance
# rf=RandomForestRegressor()
# rf.fit(X,y)
# result=permutation_importance(rf,\
# 							  X,\
# 							  y,\
# 							  n_repeats= 10,\
# 							  random_state=42,\
# 							  n_jobs=2 )
# sorted_idx=result.importances_mean.argsort()
# fig,ax=plt.subplots()
# ax.boxplot(result.importances[sorted_idx].T,\
# 		 vert=False,\
# 		 labels=X.columns[sorted_idx]  )
# ax.set_title('Permutation Importances (test set) ')
# fig.tight_layout()
# plt.show()
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
# train_x,val_x,train_y,val_y=train_test_split(X,y,random_state=1)
# my_model=RandomForestRegressor().fit(train_x,train_y)
# perm=PermutationImportance(my_model,random_state=1).fit(val_x,val_y)
# a=eli5.show_weights(perm,feature_names=val_x.columns.tolist())
# print(list(a) )



from sklearn.feature_selection import VarianceThreshold
# X=[
#     [100,1,2,3],
#     [100,4,5,6],
#     [100,7,8,9],
#     [101,11,12,13]
# ]
# sel=VarianceThreshold(1)
# sel.fit(X)   #获得方差，不需要y
# print('Variances is %s'%sel.variances_)
# print('After transform is \n%s'%sel.transform(X))
# print('The surport is %s'%sel.get_support(True))#如果为True那么返回的是被选中的特征的下标
# print('The surport is %s'%sel.get_support(False))#如果为FALSE那么返回的是布尔类型的列表，反应
