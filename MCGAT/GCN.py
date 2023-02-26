from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from MCGAT.utils import *
from MCGAT.models import GCN1, GCN
from sklearn.metrics import f1_score
import os
from MCGAT.config import Config


# def load_graph(dataset):
#     struct_edges = np.genfromtxt(dataset+"/case1.edge", dtype=np.int32)
#     sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
#     sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(900, 900), dtype=np.float32)
#     sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
#     nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
#
#     nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
#
#     return nsadj
#
# def load_data(path):
#     f = np.loadtxt(path+"/case1.feature", dtype = float)
#     l = np.loadtxt(path+"/case1.label", dtype = int)
#     test = np.loadtxt(path+"/test1.txt", dtype = int)
#     train = np.loadtxt(path+"/train1.txt", dtype = int)
#     features = sp.csr_matrix(f, dtype=np.float32)
#     features = torch.FloatTensor(np.array(features.todense()))
#
#     idx_test = test.tolist()
#     idx_train = train.tolist()
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_test = torch.LongTensor(idx_test)
#
#     label = torch.LongTensor(np.array(l))
#
#     return features, label, idx_train, idx_test


def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test


def load_graph(dataset, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = torch.cuda.is_available()

    # path = "case_study"
    #
    # sadj = load_graph(path)
    # # print(sadj)
    # features, labels, idx_train, idx_test = load_data(path)
    # # print(features, labels, idx_train, idx_test)

    # 其他数据集
    dataset = 'BlogCatalog'
    labelrate = 20
    config_file = "./config/" + str(labelrate) + str(dataset) + ".ini"
    config = Config(config_file)

    sadj, fadj = load_graph(labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)


    model = GCN1(nfeat=config.fdim,
                  nhid=config.nhid1,
                  out=config.class_num,
                  dropout=config.dropout)

    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=6e-4)


    def train(model, epochs, acc_max):
        model.train()
        optimizer.zero_grad()
        output = model(features, sadj)
        # print(output)
        # print(output[idx_train], labels[idx_train])
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        print(loss)
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1 = main_test(model, acc_max)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item()


    def main_test(model, acc_max):
        model.eval()
        output = model(features, sadj)

        acc_test = accuracy(output[idx_test], labels[idx_test])
        if acc_test >= acc_max:
            output_np = output.data.cpu().numpy()
            np.save(os.path.join('./traindata', 'gcn_output.npy'), output_np, allow_pickle=False)

        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(100):
        loss, acc_test, macro_f1 = train(model, epoch, acc_max)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))


