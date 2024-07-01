import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from torch.nn.utils.rnn import pad_sequence


class TIMITDataset(Dataset):
    def __init__(self, data_path, label_path, split='train'):

        data_file = os.path.join(data_path, f"{split}_mfccs.pickle")
        label_file = os.path.join(label_path, f"{split}_reduced_phonems.pickle")

        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.label = pickle.load(f)

        self.split = split

    def __getitem__(self, index):
        features = torch.from_numpy((self.data[index])).float()
        labels = torch.from_numpy((self.label[index])).long()

        return features, labels

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    features, labels = zip(*batch)
    seq_lengths = torch.Tensor([len(seq) for seq in features])
    padded_features = pad_sequence(features, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    return seq_lengths, padded_features, padded_labels


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    print("=== TIMIT dataset ===")
    train_loader, val_loader, test_loader = load_timit_dataset(args.data_path, args.label_path, args.batch_size)
    return device, train_loader, val_loader, test_loader


def load_timit_dataset(data_path, label_path, batch_size, subset_ratio=0.003):
    print('Loading data ...')
    train_dataset = TIMITDataset(data_path, label_path, split='train')
    val_dataset = TIMITDataset(data_path, label_path, split='val')
    test_dataset = TIMITDataset(data_path, label_path, split='test')

    # Subset of the training data
    subset_size = int(subset_ratio * len(train_dataset))
    subset_size_val = int(0.025 * len(val_dataset))
    train_dataset, _ = random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
    val_dataset, _ = random_split(val_dataset, [subset_size_val, len(val_dataset) - subset_size_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    return train_loader, val_loader, test_loader
