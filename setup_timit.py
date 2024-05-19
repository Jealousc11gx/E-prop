import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class TIMITDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split

        # 加载数据
        self.data = self.load_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        # 加载音频特征
        features_path = os.path.join(self.data_path, self.split, 'filter_banks.pickle')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        # 加载音素标签
        labels_path = os.path.join(self.data_path, self.split, 'reduced_phonems.pickle')
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)

        # 将数据和标签转换为PyTorch张量
        features = [torch.from_numpy(feat).float() for feat in features]
        labels = [torch.from_numpy(label).long() for label in labels]

        # 将数据和标签组合成元组列表
        data = [(feat, label) for feat, label in zip(features, labels)]

        return data


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    print("=== TIMIT dataset ===")
    train_loader, val_loader, test_loader = load_timit_dataset(args.data_path, args.batch_size)
    return device, train_loader, val_loader, test_loader


def load_timit_dataset(data_path, batch_size):
    # 创建数据集
    train_dataset = TIMITDataset(data_path, split='train')
    val_dataset = TIMITDataset(data_path, split='validation')
    test_dataset = TIMITDataset(data_path, split='test')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader