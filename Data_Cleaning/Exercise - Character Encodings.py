'''

'''
import pandas as pd
import numpy  as np
import chardet

np.random.seed(0)

# sample_entry=b'\xa7A\xa6n'
# print(sample_entry)
# print('data type:',type(sample_entry))
# new_entry=sample_entry.decode('big5-tw').encode('utf-8')
# print(new_entry)


with open('./fatal-police-shootings-in-the-us/PoliceKillingsUS.csv','rb') as rawData:
    result=chardet.detect(rawData.read(200000))
print(result)

police_killings= pd.read_csv('./fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding='Windows-1252')
print(police_killings.head())
police_killings.to_csv('my_file.csv',encoding='utf-8')

