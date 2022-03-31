if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Starts the experiment to reproduce index collapse evaluation"
    )
    parser.add_argument(
        "--num_parts",
        nargs="?",
        type=int,
        default=1,
        help="Determines in  how many parts the hyperparamters are split",
    )
    parser.add_argument(
        "--part",
        nargs="?",
        type=int,
        default=0,
        help="Select a part of the grid, must be smaller than num_parts",
    )
    parser.add_argument(
        "--gpu", nargs="?", type=int, required=True, help="Select used gpu"
    )

    parser.add_argument(
        "--data_dir",
        nargs="?",
        type=str,
        required=True,
        help="path to the data set (download if not available)",
    )

    args = parser.parse_args()

    assert args.part < args.num_parts

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import autoencoding as ae

import itertools


class Offset(pl.LightningModule):
    def __init__(self, offset):
        super(Offset, self).__init__()
        self.offset = offset

        self.save_hyperparameters({"encoder_offset": offset})

    def forward(self, x):
        return x - self.offset


def get_encoder(codebook_dimension, offset):
    enc = ae.nn.Lightning_Model(
        nn.Sequential(
            ae.nn.Conv2d_CBnReLU(3, 256, 4, 2, 1),
            ae.nn.Conv2d_CBnReLU(256, 256, 4, 2, 1),
            ae.nn.ResidualBlock2d(256, 256),
            ae.nn.ResidualBlock2d(256, 256),
            nn.Conv2d(256, codebook_dimension, 1),
            Offset(offset),
        )
    )

    enc.hparams.update({"encoder offset": offset})
    return enc


def get_decoder(codebook_dimension):
    return ae.nn.Lightning_Model(
        nn.Sequential(
            nn.Conv2d(codebook_dimension, 256, 1),
            ae.nn.ResidualBlock2d(256, 256),
            ae.nn.ResidualBlock2d(256, 256),
            ae.nn.ConvTransposed2d_CBnReLU(256, 256, 4, 2, 1),
            ae.nn.ConvTransposed2d_CBnReLU(256, 256, 4, 2, 1),
            nn.Conv2d(256, 3, 1),
        )
    )


def get_vq_vae_loss(
    codebook_size, codebook_dimension, codebook_offset, good_initialized
):
    m = ae.nn.Lightning_VQ_VAE_codebook_loss(codebook_dimension, codebook_size, True)

    t = m.vq.codebook
    m.vq.codebook = nn.Parameter(
        torch.cat([t[:good_initialized], t[good_initialized:] + codebook_offset], dim=0)
    )
    m.hparams.update({"codebook offset": codebook_offset})
    return m, "VQ_loss", 0.25


def get_vq_vae_ema(
    codebook_size, codebook_dimension, codebook_offset, good_initialized
):
    m = ae.nn.Lightning_VQ_VAE_ema(codebook_dimension, codebook_size, 0.99, True)

    t = m.vq.codebook
    m.vq.codebook = torch.cat(
        [t[:good_initialized], t[good_initialized:] + codebook_offset], dim=0
    )

    m.hparams.update({"codebook offset": codebook_offset})
    return m, "VQ_ema", 0.25


def get_ng_vae(codebook_size, codebook_dimension, codebook_offset, good_initialized):
    m = ae.nn.Lightning_GVQ_VAE_codebook_loss(
        codebook_dimension, codebook_size, 2, None, "linear", True
    )
    t = m.vq.codebook
    m.vq.codebook = nn.Parameter(
        torch.cat([t[:good_initialized], t[good_initialized:] + codebook_offset], dim=0)
    )

    m.hparams.update({"codebook offset": codebook_offset})
    return m, "GAS", 0.25


def main(part, num_parts, data):
    codebook_size = 512
    codebook_dimension = 2
    good_initialized = 5
    vq_models = [get_ng_vae, get_vq_vae_loss, get_vq_vae_ema]  # 3

    # 4 * 8 * 3 = 96
    dataset = ae.data.CIFAR10(data, 64)

    configurations = []

    for m in vq_models:
        configurations.extend([(m, i, 0) for i in range(7)])

    for m in vq_models:
        configurations.extend([(m, 0, i) for i in range(1,7)])

    # 12 * 3

    part_size = len(configurations) // num_parts
    start = part_size * part
    end = part_size * (part + 1) if part < num_parts - 1 else len(configurations)

    for vq_model, encoder_offset, codebook_offset in configurations[start:end]:

        encoder = get_encoder(codebook_dimension, encoder_offset)
        decoder = get_decoder(codebook_dimension)
        vq_layer, name, beta = vq_model(
            codebook_size, codebook_dimension, codebook_offset, good_initialized
        )

        model = ae.train.QuantizedAutoencoder(
            encoder, decoder, vq_layer, nn.MSELoss(), beta
        )

        name = "_".join(
            [
                name,
                "s",
                str(codebook_size),
                "d",
                str(codebook_dimension),
                "enc",
                str(encoder_offset),
                "c",
                str(codebook_offset),
            ]
        )

        logger = pl.loggers.TensorBoardLogger(
            "./_logs/index_collapse/CIFAR", name, default_hp_metric=False
        )

        trainer = pl.Trainer(
            gpus=1, logger=logger, max_epochs=50, enable_checkpointing=False
        )

        trainer.fit(model, dataset)


if __name__ == "__main__":
    main(args.part, args.num_parts, args.data_dir)
