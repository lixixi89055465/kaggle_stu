a = map(lambda a, b: a + b, [1, 2, 3], [5, 6, 7])
b = map(lambda x, y: (x ** y, x + y), [2, 4, 6], [3, 2, 1])
for i in b:
    print(i)

c = map(None, [1, 2, 3], [4, 5, 6])
print("1"*100)
print(list(c))

print("2"*100)
d=map(int, ['1', '2', 3])
print(list(d))

a= map(int,'123445666')
print(list(a))


