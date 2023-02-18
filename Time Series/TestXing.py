a, b, *c = [1, 2, 3, 4]
print(a)
print(b)
print(c)

print("0" * 100)


def myPrint(*params):
    print(params)


print("1" * 100)
myPrint(1, 2, 3)
print("2" * 100)
# myPrint(x=1, y=2, z=3)
print("3" * 100)


def myPrint2(**params):
    print(params)
    print(*params)


print("4" * 100)
myPrint2(x=1, y=2, z=3)


def myPrint3(x, y):
    print(x)
    print(y)


params = (1, 2)
print("5" * 100)
print(params)
print(*params)
myPrint3(*params)

params = {'x': 1, 'y': 2}
print("6" * 100)
myPrint3(*params)
myPrint3(**params)

