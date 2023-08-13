import pandas as pd

data = pd.DataFrame({"学号": [1, 2, 3, 4],
                     "录取": ["清华", "北大", "清华", "蓝翔"],
                     "学历": ["本科", "本科", "本科", "专科"]})
print(data)
print('0' * 100)
print(pd.get_dummies(data))
print('0' * 100)
print(pd.get_dummies(data, prefix='Hello'))

