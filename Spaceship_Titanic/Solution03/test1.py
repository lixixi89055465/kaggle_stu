from IPython.display import clear_output

clear_output()
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from lightgbm import LGBMClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier
import time
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
submission = pd.read_csv('../data/sample_submission.csv')

RANDOM_STATE = 12
FOLDS = 5
STRATEGY = 'median'
'''
- `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group 
    the passenger is travelling with and pp is their number within the group. People in a group are often 
    family members, but not always.
- `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
- `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration 
    of the voyage. Passengers in cryosleep are confined to their cabins.
- `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be 
    either P for Port or S for Starboard.
- `Destination` - The planet the passenger will be debarking to.
- `Age` - The age of the passenger.
- `VIP` - Whether the passenger has paid for special VIP service during the voyage.
- `RoomService`, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the 
    Spaceship Titanic's many luxury amenities.
- `Name` - The first and last names of the passenger.
- `Transported` - Whether the passenger was transported to another dimension. This is the target, the column 
    you are trying to predict.

'''
print(train.head())
print(f'\033[94mNumber of rows in train data:{train.shape[0]}')
print(f'\033[94mNumber of columns in train data:{train.shape[1]}')
print(f'\033[94mnumber of values in train data:{train.count().sum()}')
print(f'\033[94mnumber of missing values in train data:{sum(train.isna().sum())}')
print(train.isna().sum().sort_values(ascending=False))
print(train.describe())

print(test.head())
print(f'\033[94mNumber of rows in train data:{test.shape[0]}')
print(f'\033[94mNumber of columns in train data:{test.shape[1]}')
print(f'\033[94mnumber of values in train data:{test.count().sum()}')
print(f'\033[94mnumber of missing values in train data:{sum(test.isna().sum())}')

print(test.isna().sum().sort_values(ascending=False))

print(test.describe())
print(submission.head())

train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)
TARGET = 'PassengerId'
FEATURES = [col for col in train.columns if col != TARGET]
RANGE_STATE = 12

# train.iloc[:,:-1].describe().T.sort_values(by='std',ascending=False)\
#     .style.background_gradient(cmap='GnBu')\
#     .bar(subset=['max'],color='#BB0000')\
#     .bar(subset=['mean'],color='green')
test_null = pd.DataFrame(test.isna().sum())
test_null=testf'd's'la'
