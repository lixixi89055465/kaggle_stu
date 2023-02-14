seed = 123
import pandas as pd
import numpy as np
from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#
df = load_dataset('tips').drop(['tip', 'sex'], axis=1).sample(n=5, random_state=seed)
# df = load_dataset('tips').sample(n=5, random_state=seed)
#
df.iloc[[1, 2, 4], [2, 4]] = np.nan
print(df)
# 划分数据
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['total_bill', 'size']),
                                                    df['total_bill'],
                                                    test_size=.2,
                                                    random_state=seed)

print("0" * 100)

imputer = SimpleImputer(strategy='constant', fill_value='missing')
X_train_imputed = imputer.fit_transform(X_train)
#
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = encoder.fit_transform(X_train)
#
X_test_imputed = imputer.transform(X_train_imputed)
X_test_encoded = encoder.transform(X_test_imputed)
# 1.pipeline
print("1" * 200)
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
a = pd.DataFrame(pipe.fit_transform(X_train), columns=pipe['encoder'].get_feature_names_out(X_train.columns))
b = pd.DataFrame(pipe.transform(X_test), columns=pipe['encoder'].get_feature_names_out(X_train.columns))
print(a)
print(b)
print("2" * 100)
#
imputer = SimpleImputer(strategy='constant', fill_value='missing')
X_train_imputed = imputer.fit_transform(X_train)
#
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = encoder.fit_transform(X_train_imputed)
model = LinearRegression()
model.fit(X_train_encoded, y_train)
#
y_train_pred = model.predict(X_train_encoded)
print(y_train_pred)

X_test_imputed = imputer.transform(X_test)
X_test_encoded = encoder.transform(X_test_imputed)
print("3" * 100)
y_test_pred = model.predict(X_test_encoded)
print("4" * 100)
print(y_test_pred)
print("3" * 200)
pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ('model', LinearRegression())
])
pipe.fit(X_train, y_train)
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)
print(y_train_pred)
print(y_test_pred)
print("0" * 100)
print("4" * 200)
train_test_split(df.drop(columns=['total_bill']))

# 划分数据
X_train, X_test, y_train, y_test = \
    train_test_split(df.drop(columns=['total_bill']),
                     df['total_bill'],
                     test_size=.2,
                     random_state=seed)
categorical = list(X_train.select_dtypes('category').columns)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['total_bill']),
                                                    df['total_bill'],
                                                    test_size=.2,
                                                    random_state=seed)

# 定义分类列
categorical = list(X_train.select_dtypes('category').columns)
print(categorical)
numerical = X_train.select_dtypes('number').columns
print(numerical)

print("5" * 200)
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipe, categorical)],
    remainder='passthrough')
preprocessor.fit(X_train)
print("0" * 100)
#
cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names(categorical)
columns = np.append(cat_columns, numerical)
#
print("1" * 100)
print(pd.DataFrame(preprocessor.transform(X_train), columns=columns))
print(pd.DataFrame(preprocessor.transform(X_test), columns=columns))
print('2' * 100)

