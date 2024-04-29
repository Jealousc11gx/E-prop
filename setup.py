# this file is used to prepare the data
# Author： Chen Linliang
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc


class TIMITDataset(Dataset):
    def __init__(self, X, y=None, n_t=11, n_o=39):
        """
        初始化数据集。

        参数:
        X -- 特征数据，形状为 [num_samples, 11, 39]
        y -- 标签数据，形状为 [num_samples]，每个样本一个整数标签
        n_t -- 时间步数量，这里为11
        n_o -- 类别数量，这里为39
        """
        self.data = torch.from_numpy(X).float().view(-1, 11, 39)
        if y is not None:
            y = y.astype(np.int64)
            self.label = self.onehot_convertor(torch.from_numpy(y).long(), n_t, n_o)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


    def onehot_convertor(self, labels, n_t, n_o):
        """
        将整数形式的标签转换为独热码形式。
        """
        n_b = labels.size(0)
        one_hot_labels = torch.zeros(n_b, n_t, n_o)
        for b in range(n_b):
            one_hot_labels[b, :, labels[b]] = 1
        return one_hot_labels


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    print("=== TIMIT dataset...")
    (train_loader, test_loader) = load_timit_dataset()
#   print("Training set length: " + str(args.full_train_len))
#   print("Test set length: " + str(args.full_test_len))
    return device, train_loader, test_loader


def load_timit_dataset(VAL_RATIO=0.10, BATCH_SIZE=256):
    print('Loading data ...')

    # 加载原始文件
    data_root = 'e:/TIMIT/timit_11/'
    train = np.load(data_root + 'train_11.npy')
    train_label = np.load(data_root + 'train_label_11.npy')
    # print('Size of all data: {}'.format(train.shape))
    #  print('Size of testing data: {}'.format(train.shape * VAL_RATIO))

    # 设置训练集和测试集比例 对于超过百万级别的大模型，其测试集占2.5%合适 这里设置测试集为百分之五
    train_size = int(len(train) * (1 - VAL_RATIO))  # 90%
    train_x, train_y = train[:train_size], train_label[:train_size]
    test_x, test_y = train[train_size:], train_label[train_size:]
    # print('Size of training set: {}'.format(train_x.shape))
    # print('Size of test set: {}'.format(test_x.shape))

    # 设置dataset和dataloader
    train_set = TIMITDataset(train_x, train_y)
    test_set = TIMITDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    del train, train_label, train_x, train_y
    gc.collect()

    return train_loader, test_loader
