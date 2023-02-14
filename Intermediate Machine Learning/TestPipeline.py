from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# 划分训练集与测试集，并分层抽样
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, random_state=0)

# 特征变量标准化
# 由于支持向量机可能受特征变量取值范围影响，训练集与测试集的特征变量标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
#
svm = SVC()
svm.fit(X_train, y_train)
print('single svm score:', svm.score(X_test, y_test))
print("0" * 100)
param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.001, 0.01, 0.1, 1, 10, 100]
}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
model = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=kfold)
model.fit(X_train_s, y_train)
print('网络搜索预测准确度:', model.score(X_test_s, y_test))
print('最优超参数:', model.best_params_)
print("2" * 100)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
pipe.fit(X_train_s, y_train)
print("3" * 100)
print(pipe.score(X_test_s, y_test))
print("4" * 100)
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', model)
# ])

print("5" * 100)
# print(pipe.fit(X_train, y_train))
print("6" * 100)
# print(pipe.score(X_test, y_test))
print("7" * 100)
# print(pipe.named_steps['model'].best_params_)
# print("8" * 100)
# print(pipe.named_steps['model'])
print("9" * 100)
print(pipe.steps)
print("0" * 100)
pipe_short = make_pipeline(StandardScaler(), SVC())
print(pipe_short.steps)

print("1" * 100)
print(pipe_short.named_steps['svc'].support_vectors_)
