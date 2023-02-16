import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

'''
1 .使用Sklearn的LabelBinarizer来进行one-hot
'''

# 创建特征
feature = np.array([['Texas'],
                    ['Cliforniya'],
                    ['Texas'],
                    ['Delaware'],
                    ['Texas']])
#

# 实例化编码器
one_hot = LabelBinarizer()
display(one_hot.fit_transform(feature))
display(one_hot.classes_)
