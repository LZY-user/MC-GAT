import numpy as np

num1 = []

data = open("test1.txt")
for line in data.readlines():
    line = int(line)
    num1.append(line)
print(num1)
data = open("train1.txt")
for line in data.readlines():
    line = int(line)
    num1.append(line)


arr = []
for i in range(900):
    arr.append(i)
# print(arr)

diff = list(set(arr) - set(num1))
filename = 'val1.txt'
with open(filename,'a+') as f:
    for i in range(240):
        f.write(str(diff[i]))
        f.write("\n")



num1 = []

data = open("test2.txt")
for line in data.readlines():
    line = int(line)
    num1.append(line)
print(num1)
data = open("train2.txt")
for line in data.readlines():
    line = int(line)
    num1.append(line)


arr = []
for i in range(900):
    arr.append(i)
# print(arr)

diff = list(set(arr) - set(num1))
filename = 'val2.txt'
with open(filename,'a+') as f:
    for i in range(240):
        f.write(str(diff[i]))
        f.write("\n")