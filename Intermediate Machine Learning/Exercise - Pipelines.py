import pandas as pd
from sklearn.model_selection import train_test_split

#
X_full = pd.read_csv('../input/home-input-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-input-for-ml-course/train.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Cardinality
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype == 'object' and X_train_full[cname].nunique() < 10]
print(categorical_cols)
numerical_cols = [cname for cname in X_train_full.columns
                  if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
print("0" * 100)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#
numerical_transformer = SimpleImputer(strategy='constant')
#
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
model = RandomForestRegressor(n_estimators=100, random_state=0)
clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])
clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
print('mae')
print(mean_absolute_error(y_valid, y_pred=preds))

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
model = RandomForestRegressor(random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])

# Preprocessing of training input, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation input, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


