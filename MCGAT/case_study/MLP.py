import numpy as np
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


data = open("case1.feature")

zns = []
for line in data.readlines():
    nums = line.split(" ")
    nums = np.array(nums)
    y = nums.astype(np.float)
    zns.append(y)

ts = torch.tensor(zns).unsqueeze(0)
ts = ts.float()
print(ts.shape)

#label
data = open("case1.label")

zns = []

for line in data.readlines():
    line = int(line)
    s = [0,0,0]
    s[line] = 1
    nums = np.array(s)
    y = nums.astype(np.float)
    zns.append(y)

labels = torch.tensor(zns).unsqueeze(0)
labels = labels.float()
print(labels.shape)


net = Net(50, 128, 3)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(200):
    prediction = net(ts)
    loss = loss_func(prediction, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def accuracy(output, labels):
    preds = output.argmax(axis=2)
    correct = 0
    i=0
    for pred in preds[0]:
        # print(pred,labels[0][i])
        if labels[0][i][pred] == 1:
            correct = correct + 1
        i = i +1
    return (float)(correct) / labels.shape[1]


net.eval()
prediction = net(ts)

acc = accuracy(prediction, labels)

print(acc)
