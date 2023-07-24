import numpy as np
import pandas as pd

boolean = [True, False]
gender = ['男', '女']
color = ['white', 'black', 'yellow']
data = pd.DataFrame({
    'height': np.random.randint(150, 190, 100),
    'weight': np.random.randint(40, 90, 100),
    'smoker': [boolean[x] for x in np.random.randint(0, 2, 100)],
    'gender': [gender[x] for x in np.random.randint(0, 2, 100)],
    'age': np.random.randint(15, 90, 100),
    'color': [color[x] for x in np.random.randint(0, len(color), 100)]
})


# 使用字典进行映射
# data['gender'] = data['gender'].map({'男': 1, '女': 0})
def gender_map(x):
    return i if x == '男' else 0


# data['gender_num'] = data['gender'].map(gender_map)
# print('2' * 100)
# print(data)


# 以元组的方式传入额外的参数
def apply_age(x, bias):
    return x + bias


data['age'] = data['age'].apply(apply_age, args=(-300,))
print('3' * 100)
print(data)

print(data[["height", "weight", "age"]].apply(np.sum, axis=0))
data[["height", "weight", "age"]].apply(np.log, axis=0)

print('4' * 100)
print(data[["height", "weight", "age"]].apply(np.log, axis=0))
