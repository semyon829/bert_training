import pandas as pd
import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):
    def __init__(self, meta_path, dataset_type, train, transforms, meta=None):
        self.dataset_type = dataset_type
        self.train = train
        if not meta is None:
            self.meta = meta
        else:
            self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta["train_val_test_split"] == self.dataset_type].reset_index(drop=True)
        self.transforms = transforms

        self.premises = self.meta['premise'].values
        self.hypotheses = self.meta['hypothesis'].values
        self.labels = self.meta['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        return (premise, hypothesis), label


def __load__(*args, **kwargs):
    return NLIDataset(*args, **kwargs)