import numpy as np

arr1 = np.array([1, 2, 3])
print(arr1.shape)  # (3,)代表arr1是一个包含3个元素的一维数组  可以把它看作是一个列向量
arr2 = np.array([4, 5, 6])
print(np.c_[arr1, arr2])  # 行数相同

print("1"*100)
arr3 = np.array([[1, 2], [4, 5], [7, 8]])
print(np.c_[arr1, arr2, arr3])  # 行数相同
print("2"*100)

arr4 = np.random.randint(0, 10, (5, 3))
print(np.c_[np.ones(5), arr4])  # 行数相同
print("3"*100)

arr1 = np.array([1, 2, 3, 4])
print(arr1.shape)  # (3,)代表arr1是一个包含4个元素的一维数组  可以把它看作是一个列向量
arr2 = np.array([4, 5, 6, 7, 8])
print(np.r_[arr1, arr2])  # 列数相同

arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr4 = np.random.randint(0, 10, (5, 3))
print(np.r_[arr3, arr4])  # 列数相同
