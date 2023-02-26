from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MCGAT.utils import *
from MCGAT.models import SFGCN, SFGAT, GAT
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from MCGAT.config import Config
#MCGAT
###################

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = 'uai'
    labelrate = 20
    config_file = "./config/" + str(labelrate) + str(dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

   
    sadj, fadj = load_graph(labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)

    # print(features)
    
    model = SFGAT(nfeat = config.fdim,
              nhid1 = config.nhid1,
              nhid2 = config.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout,
                  alpha = config.alpha)

    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train(model, epochs, acc_max):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb= model(features, sadj, fadj)
        # print(att)
        loss_class =  F.nll_loss(output[idx_train], labels[idx_train])
        loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
        loss_com = common_loss(com1,com2)
        print(loss_class,loss_com,loss_dep)
        loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        print(loss)
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model, acc_max)

        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test

    def main_test(model, acc_max):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        # print(output.shape)
        # 保存
        # if acc_test >= acc_max:
        #     output_np = output.data.cpu().numpy()
        #     np.save(os.path.join('./traindata', 'AMgat_output.npy'), output_np, allow_pickle=False)

        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb
    
    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch, acc_max)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))


