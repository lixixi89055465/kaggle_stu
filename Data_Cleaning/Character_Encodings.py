'''

'''
import pandas as pd
import numpy as np
import chardet

np.random.seed(0)

# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
print(type(before))


after=before.encode('utf-8',errors='replace')
print(type(after))
print(after)
print(after.decode('utf-8'))
# print(after.decode('ascii'))
# before = "This is the euro symbol: €"
# after=before.encode('ascii',errors='replace')
# print(after.decode('ascii'))

# kickstarter_2016 = pd.read_csv("./Kickstarter_Projects/ks-projects-201612.csv")
# with open('./Kickstarter_Projects/ks-projects-201801.csv','rb') as rawdata:
#     result=chardet.detect(rawdata.read(10000))
#
# print(result)
#
# print('1'*100)
# kickstarter_2016=pd.read_csv('./Kickstarter_Projects/ks-projects-201612.csv',encoding='Windows-1252')
# print(kickstarter_2016.head())
#
# kickstarter_2016.to_csv('./Kickstarter_Projects/kickstarter_2018_utf8.csv')
