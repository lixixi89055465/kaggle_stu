# -*- coding: utf-8 -*-
# @Time : 2024/10/21 22:47
# @Author : nanji
# @Site : 
# @File : testCatboost02.py
# @Software: PyCharm 
# @Comment :

from catboost.datasets import titanic
import numpy as np

train_df, test_df = titanic()
print(train_df.head())

null_value_stats = train_df.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]

train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

X = train_df.drop('Survived', axis=1)
y = train_df.Survived
print(X.dtypes)
categorical_features_indices = np.where(X.dtypes != float)[0]
print(categorical_features_indices)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)

X_test = test_df
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

model = CatBoostClassifier(
    custom_loss=[metrics.Accuracy()],
    random_state=42,
    logging_level='Silent'
)
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    plot=True
)
print('1' * 100)

cv_params = model.get_params()
cv_params.update({
    'loss_function': metrics.Logloss()
})

cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)

print('Precise validation accuracy score:{}'.format(np.max(cv_data['test-Accuracy-mean'])))

predictions = model.predict(X_test)
prediction_probs = model.predict_proba(X_test)
print(predictions[:10])
print(prediction_probs[:10])

model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')
model_without_seed.fit(X, y, cat_features=categorical_features_indices)

print('Random seed assigned for this model :{}'.format(model_without_seed.random_seed_))
params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': metrics.Accuracy(),
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': False
}
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validation_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validation_pool)
best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validation_pool)
print('Simple model validation accuracy:{:.4}'.format(accuracy_score(y_validation, model.predict(X_validation))))
print('')
print('Best model validation accuracy:{:.4}'.format(accuracy_score(y_validation, best_model.predict(X_validation))))
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validation_pool)
earlystop_params = params.copy()
earlystop_params.update({
    'od_type': 'Iter',
    'od_wait': 40
})
earlystop_model = CatBoostClassifier(**earlystop_params)
earlystop_model.fit(train_pool, eval_set=validation_pool)

print('Simple model tree count:{}'.format(model.tree_count_))
print('Simple model validation accuracy:{:.4}'.format(
    accuracy_score(y_validation, model.predict(X_validation))
))

print('Early-stopped model tree count:{}'.format(earlystop_model.tree_count_))
print('Early-stopped model validation accuracy :{:.4}'.format(
    accuracy_score(y_validation, earlystop_model.predict(X_validation))
))

current_params = params.copy()
current_params.update({
    'iterations': 10
})
model = CatBoostClassifier(**current_params).fit(X_train, y_train, categorical_features_indices)
baseline = model.predict(X_train, prediction_type='RawFormulaVal')
# Fit new model
model.fit(X_train, y_train, categorical_features_indices, baseline=baseline)

params_with_snappshot = params.copy()
params_with_snappshot.update({
    'iterations': 5,
    'learning_rate': 0.5,
    'logging_level': 'Verbose'
})
model = CatBoostClassifier(**params_with_snappshot).fit(train_pool, eval_set=validation_pool,
                                                        save_snapshot=True)
params_with_snappshot.update({
    'iteration': 10,
    'learning_rate': 0.1
})
model = CatBoostClassifier(
    **params_with_snappshot
).fit(train_pool, eval_set=validation_pool, save_snapshot=True)

