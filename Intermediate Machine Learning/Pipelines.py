import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
y = data.Price
X = data.drop(['Price'], axis=1)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype == 'object' and X_train_full[cname].nunique() < 10]

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
print("1" * 100)
print(X_train.head())
print("2" * 100)
print(X_train.isnull().sum())
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical input
numerical_transformer = SimpleImputer(strategy='constant')
# Preprocessing for categorical input

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical input
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('model', model)])
# Preprocessing of training input, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation input, get predictions
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, y_pred=preds)
print('mae:', score)
