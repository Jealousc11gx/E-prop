# this file is used to prepare the data
# Author： Chen Linliang
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float().view(-1, 11, 39)
        if y is not None:
            y = y.astype(np.int64)
            self.label = torch.from_numpy(y).long()
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    print("=== TIMIT dataset ===")
    (train_loader, val_loader, test_loader) = load_timit_dataset()
    return device, train_loader, val_loader, test_loader


def load_timit_dataset(TRAIN_RATIO=0.9, VAL_RATIO=0.05,  BATCH_SIZE=128):
    print('Loading data ...')

    # 加载原始文件
    data_root = 'e:/TIMIT/timit_11/'
    train = np.load(data_root + 'train_11.npy')
    train_label = np.load(data_root + 'train_label_11.npy')

    # 设置训练集和测试集比例 对于超过百万级别的大模型，其测试集占2.5%合适 这里设置测试集为百分之五
    train_size = int(len(train) * TRAIN_RATIO)  # 90%
    val_size = int(len(train) * VAL_RATIO)

    # # 生成随机索引 一开始的数据是规则分布的
    # permutation = np.random.permutation(train.shape[0])
    #
    # # 使用随机索引重新排列数据和标签 打破顺序bias
    # train = train[permutation]
    # train_label = train_label[permutation]

    train_x, train_y = train[:train_size], train_label[:train_size]
    val_x, val_y = train[train_size:train_size+val_size], train_label[train_size:train_size+val_size]
    test_x, test_y = train[train_size+val_size:], train_label[train_size+val_size:]

    # print(f"size of the train set is {train_x.shape}")
    # print(f"size of the validation set is {val_x.shape}")
    # print(f"size of the test set is {test_x.shape}")

    # 设置dataset和dataloader
    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    test_set = TIMITDataset(test_x, test_y)
    # print(f"datashape is {train_set.data.shape}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 内存回收
    del train, train_label, train_x, train_y, val_x, val_y, test_x, test_y
    gc.collect()

    return train_loader, val_loader, test_loader
