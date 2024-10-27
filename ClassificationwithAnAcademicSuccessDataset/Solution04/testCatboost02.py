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
    'iterations': 10,
    'learning_rate': 0.1
})
model = (CatBoostClassifier(**params_with_snappshot).
         fit(train_pool, eval_set=validation_pool, save_snapshot=True))


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)
            if weights is not None:
                assert len(weights) == len(approxes)
            result.append((der1, der2))
        return result


model = CatBoostClassifier(
    iterations=10,
    random_seed=42,
    loss_function=LoglossObjective(),
    eval_metric=metrics.Logloss()
)
# Fit model
model.fit(train_pool)
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')


class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (target[i] * approx[i] - np.log(1 + np.exp(approx[i])))
        return error_sum, weight_sum


class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (target[i] * approx[i] - np.log(1 + np.exp(approx[i])))
        return error_sum, weight_sum


model = CatBoostClassifier(
    iterations=10,
    random_seed=42,
    loss_function=metrics.Logloss(),
    eval_metric=LoglossMetric()
)
# fit model
model.fit(train_pool)
# Only prediction_type = 'RawFormulaVal' is allowed with custom 'loss_function'
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')

model = CatBoostClassifier(iterations=10, random_seed=42, logging_level='Silent').fit(train_pool)
ntree_start, ntree_end, eval_period = 3, 9, 2
predictions_iterator = model.staged_predict(validation_pool, 'Probability', ntree_start, ntree_end, eval_period)
for preds, tree_count in zip(predictions_iterator, range(ntree_start, ntree_end, eval_period)):
    print('First class probabilities using the first {} trees:{}'.format(tree_count, preds[:5, 1]))

model = CatBoostClassifier(iterations=50, random_seed=42, logging_level='Silent').fit(train_pool)
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {'feature_names': feature_names, 'feature_importance': feature_importances}
fi_df = pd.DataFrame(data)
fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
# Add chart labels

plt.title('FEATURE IMPORTANCE')
plt.xlabel('FEATUre importance')
plt.ylabel('feature names')

model = CatBoostClassifier(iterations=50, random_seed=42, logging_level='Silent').fit(train_pool)
eval_metrics = model.eval_metrics(validation_pool, [metrics.AUC()], plot=True)

print(eval_metrics['AUC'][:6])

model1 = CatBoostClassifier(iterations=100, depth=1, train_dir='model_depth_1/', logging_level='Silent')
model1.fit(train_pool, eval_set=validation_pool)
model2 = CatBoostClassifier(iterations=100, depth=5, train_dir='model_depth_5/', logging_level='Silent')
model2.fit(train_pool, eval_set=validation_pool)

from catboost import MetricVisualizer

widget = MetricVisualizer(['model_depth_1', 'model_depth_5/'])
widget.start()

### 3.11
model = CatBoostClassifier(iterations=10, random_seed=42, logging_level='Silent').fit(train_pool)
model.save_model('catboost_model.dump')
model = CatBoostClassifier()
model.load_model('catboost_model.dump')

import hyperopt


def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        learning_rate=params['learning_rate'],
        iterations=500,
        eval_metric=metrics.Accuracy(),
        random_seed=42,
        verbose=False,
        loss_function=metrics.Logloss()
    )
    cv_data = cv(
        Pool(X, y, cat_features=categorical_features_indices),
        model.get_params(),
        logging_level='Silent'
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])
    return 1 - best_accuracy  # as hyperopt minimises



from numpy.random import RandomState
params_space={
    'l2_leaf_reg':hyperopt.hp.qloguniform('l2_leaf_reg',0,2,1),
    'learning_rate':hyperopt.hp.uniform('learning_rate',1e-3,5e-1)
}
trials=hyperopt.Trials()
best=hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials,
    #rstate=RandomState(123)
)
