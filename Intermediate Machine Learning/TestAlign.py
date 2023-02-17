import pandas as pd

df = pd.DataFrame(
    [[1, 2, 3, 4],
     [6, 7, 8, 9]],
    columns=["D", "B", "E", "A"],
    index=[1, 2]
)

other = pd.DataFrame(
    [[10, 20, 30, 40], [60, 70, 80, 90], [600, 700, 800, 900]],
    columns=["A", "B", "C", "D"],
    index=[2, 3, 4],
)
print("2" * 100)
print(df)
print("3" * 100)
print(other)

print("4" * 100)
left, right = df.align(other, join='left', axis=1)
print(left)
print("5" * 100)
print(right)
print("6" * 100)
left, right = df.align(other, join='left', axis=1)
print(left)
print("7" * 100)
print(right)
print("8" * 100)
left, right = df.align(other, join='left', axis=0)
print(left)
print("9" * 100)
print(right)
print("10" * 100)
left, right = df.align(other, join='left', axis=None)
print(left)
print(right)

