from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd

test_list = ['05db9164', '68fd1e64', '05db9164', '8cf07265', '05db9164',
             '68fd1e64', '5bfa8ab5', '5a9ed9b0', '05db9164', '9a89b36c',
             '68fd1e64', '8cf07265', '05db9164', '68fd1e64', '5a9ed9b0',
             '68fd1e64', '5a9ed9b0', '05db9164', '05db9164', '2d4ea12b']

lbe = LabelEncoder()
lbe_res = lbe.fit_transform(test_list)
print('0' * 100)
print(len(lbe_res))
print(lbe_res)
print('1' * 100)
print(lbe.classes_)
print('2' * 100)

print(lbe_res)
print('3' * 100)
print(lbe.inverse_transform([5, 2, 1]))
encoder_x = LabelEncoder()
data = ['b', 'c', 'e', 'f']
from sklearn import preprocessing

enc = preprocessing.LabelEncoder()
enc = enc.fit(data)
print('4' * 100)
data = enc.transform(data)
print(data)
print('5'*100)
print(enc.inverse_transform([0, 2, 1]))
