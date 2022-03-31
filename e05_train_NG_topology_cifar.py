if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trains a single Neural-Gas VAE model with a lifetime value"
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

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import autoencoding as ae


def get_encoder(codebook_dimension):
    return ae.nn.Lightning_Model(
        nn.Sequential(
            ae.nn.Conv2d_CBnReLU(3, 256, 4, 2, 1),
            ae.nn.Conv2d_CBnReLU(256, 256, 4, 2, 1),
            ae.nn.ResidualBlock2d(256, 256),
            ae.nn.ResidualBlock2d(256, 256),
            nn.Conv2d(256, codebook_dimension, 1),
        )
    )


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


def get_ng_vae(codebook_size, codebook_dimension):
    return (
        ae.nn.Lightning_GVQ_VAE_codebook_loss(
            codebook_dimension, codebook_size, 1, 2, "linear", True
        ),
        "GAS",
        0.25,
    )


def get_model(codebook_size, codebook_dimension):
    encoder = get_encoder(codebook_dimension)
    decoder = get_decoder(codebook_dimension)
    vq_layer, name, beta = get_ng_vae(codebook_size, codebook_dimension)
    return ae.train.QuantizedAutoencoder(encoder, decoder, vq_layer, nn.MSELoss(), 0.5)


def main(data):

    dataset = ae.data.CIFAR10(data, 64)

    name = "NG"
    codebook_size = 512
    codebook_dimension = 2
    model = get_model(codebook_size, codebook_dimension)

    name = "_".join([name, "s", str(codebook_size), "d", str(codebook_dimension)])

    logger = pl.loggers.TensorBoardLogger(
        "./_logs/ng/CIFAR", name, default_hp_metric=False
    )

    callbacks = [
        ModelCheckpoint(
            monitor="best/MSELoss",
            dirpath="./_checkpoints/neural_gas/CIFAR/checkpint",
            save_top_k=3,
        )
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        gpus=1,
        logger=logger,
        max_epochs=100,
        enable_checkpointing=True,
    )

    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
