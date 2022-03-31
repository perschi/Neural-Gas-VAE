import torch
import torchaudio

import pytorch_lightning as pl

from torch.utils.data import DataLoader


class Speechcommands(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size=32, train_transform=None, test_transform=None
    ):
        super(Speechcommands, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        # download
        torchaudio.datasets.SPEECHCOMMANDS(root=self.data_dir, download=True)
        self.labels = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]

        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}

    @staticmethod
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.0
        )
        return batch.permute(0, 2, 1)

    @staticmethod
    def create_collate(transform, label2id):
        if transform is None:
            transform = lambda x: x

        def collate(batch):
            # A data tuple has the form:
            # waveform, sample_rate, label, speaker_id, utterance_number

            tensors, sample_rates, labels, speaker_ids, utterance_numbers = (
                [],
                [],
                [],
                [],
                [],
            )

            # Gather in lists, and encode labels as indices
            for waveform, sample_rate, label, speaker_id, utterance_number in batch:
                tensors += [transform(waveform)]
                labels.append(label2id[label])
                sample_rates.append(sample_rate)
                speaker_ids.append(speaker_id)
                utterance_numbers.append(utterance_number)
            # Group the list of tensors into a batched tensor
            tensors = Speechcommands.pad_sequence(tensors)
            labels = torch.tensor(labels)
            sample_rates = sample_rates
            speaker_ids = speaker_ids
            utterance_numbers = utterance_numbers

            return tensors, labels, sample_rates, speaker_ids, utterance_numbers

        return collate

    def train_dataloader(self):
        return DataLoader(
            torchaudio.datasets.SPEECHCOMMANDS(root=self.data_dir, subset="training"),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self.create_collate(self.train_transform, self.label2id),
        )

    def val_dataloader(self):
        return DataLoader(
            torchaudio.datasets.SPEECHCOMMANDS(root=self.data_dir, subset="validation"),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.create_collate(self.test_transform, self.label2id),
        )

    def test_dataloader(self):
        return DataLoader(
            torchaudio.datasets.SPEECHCOMMANDS(root=self.data_dir, subset="testing"),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.create_collate(self.test_transform, self.label2id),
        )
