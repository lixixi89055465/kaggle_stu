import numpy as np
import pandas as pd

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print(train_df.sample(3))
print(train_df.columns)
print(train_df.shape)
train_missing = pd.DataFrame(train_df.isna().sum()).sort_values(by=0, ascending=False)
test_missing = pd.DataFrame(test_df.isna().sum()).sort_values(by=0, ascending=False)
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# fig = make_subplots(1,2,column_titles=['train','test'],x_title='Missing Values')
#
# fig.add_trace(go.Bar(x=train_missing[0],y=train_missing.index,orientation="h",
#                      marker=dict(color=[n for n in range(12)])
#                     ),1,1
#              )
# fig.add_trace(go.Bar(x=test_missing[0],y=test_missing.index,orientation="h",
#                      marker=dict(color=[n for n in range(12)])
#                     ),1,2
#              )
# fig.update_layout(showlegend=False, title_text="Missing Values In Train&Test Set", title_x=0.5)
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count()) * 100
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return (tt)

missing_info=missing_data(train_df)
print(missing_info)
missing_info.query('Types="object"')
