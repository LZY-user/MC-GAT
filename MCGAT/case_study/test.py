import numpy as np

data = open("case1.label")
num1 = []
num2 = []
num3 = []
i = 0
for line in data.readlines():
    line = int(line)
    if line == 0:
        num1.append(i)
    elif line == 1:
        num2.append(i)
    else:
        num3.append(i)
    i += 1
# print(num1)
# print(num2)
# print(num3)
list1 = np.random.choice(num1, 220, replace=False)
filename = 'test1.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train1.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")

list1 = np.random.choice(num2, 220, replace=False)
filename = 'test1.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train1.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")

list1 = np.random.choice(num3, 220, replace=False)
filename = 'test1.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train1.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")



data = open("case2.label")
num1 = []
num2 = []
num3 = []
i = 0
for line in data.readlines():
    line = int(line)
    if line == 0:
        num1.append(i)
    elif line == 1:
        num2.append(i)
    else:
        num3.append(i)
    i += 1

# print(len(num1))
list1 = np.random.choice(num1, 220, replace=False)
filename = 'test2.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train2.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")

list1 = np.random.choice(num2, 220, replace=False)
filename = 'test2.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train2.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")

list1 = np.random.choice(num3, 220, replace=False)
filename = 'test2.txt'
with open(filename,'a+') as f:
    for i in range(200):
        f.write(str(list1[i]))
        f.write("\n")
filename = 'train2.txt'
with open(filename, 'a+') as f:
    for i in range(20):
        f.write(str(list1[i+200]))
        f.write("\n")