# https://zhuanlan.zhihu.com/p/101284491
import pandas as pd
import numpy as np

company = ["A", "B", "C"]
data = pd.DataFrame({
    "company": [company[x] for x in np.random.randint(0, len(company), 10)],
    "salary": np.random.randint(5, 50, 10),
    "age": np.random.randint(15, 50, 10)
})
print(data)
print(data.columns)
print('1' * 100)
group = data.groupby('company')
print(list(group))
print('2' * 100)
print(data.groupby('company').agg('mean'))
print('3' * 100)
print(data.groupby('company').agg('min'))
print('4' * 100)
print(data.groupby('company').agg({'salary': 'median', 'age': 'mean'}))

print('5' * 100)
avg_salary_dict = data.groupby('company')['salary'].mean().to_dict()
print(data['company'].map(avg_salary_dict))
print('6' * 100)
data['avg_salary'] = data.groupby('company')['salary'].transform('mean')
print(data)
print('7' * 100)
df = data.sort_values(by='age', ascending=True)
print(df)
print(df.iloc[-1, :])
print('8' * 100)


def get_oldest_staff(x):
    df = x.sort_values(by='age', ascending=True)
    return df.iloc[-1, :]


print('9' * 100)
oldest_staff = data.groupby('company', as_index=False).apply(get_oldest_staff)
print(oldest_staff)
