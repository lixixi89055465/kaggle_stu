from IPython.display import clear_output

clear_output()
import numpy as np
import pandas as pd
import seaborn as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from lightgdm import LGBMClassifier
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
