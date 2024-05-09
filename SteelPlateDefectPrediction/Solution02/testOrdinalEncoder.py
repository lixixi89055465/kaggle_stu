# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 下午10:39
# @Author  : nanji
# @Site    :https://blog.csdn.net/WHYbeHERE/article/details/135527212
# @File    : testOrdinalEncoder.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
from category_encoders import OrdinalEncoder

data = {
	'Category': ['low', 'Medium', 'High', 'Low', 'Medium', 'High']
}
df = pd.DataFrame(data)

enc = OrdinalEncoder()
enc.fit(df)
df_encoded = enc.fit_transform(df)
print(df_encoded)

r1=enc.inverse_transform(df_encoded)
print('0'*100)
print(r1)
