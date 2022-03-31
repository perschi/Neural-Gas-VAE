import torch
import torchvision
import pytorch_lightning as pl


class CIFAR10(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = torchvision.transforms.ToTensor()

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, download=True)

    def setup(self, stage=None):
        self.train_data = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, download=False, transform=self.transforms
        )
        self.test_data = torchvision.datasets.CIFAR10(
            self.data_dir, train=False, download=False, transform=self.transforms
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, shuffle=True, batch_size=self.batch_size, drop_last=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, shuffle=False, batch_size=self.batch_size, drop_last=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, shuffle=False, batch_size=self.batch_size, drop_last=False
        )
