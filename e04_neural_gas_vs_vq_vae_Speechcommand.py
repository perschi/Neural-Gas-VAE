if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start the comparison in Speechcommand"
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


def get_encoder(codebook_dimension):
    return ae.nn.Lightning_Model(
        nn.Sequential(
            ae.nn.Conv1d_CBnReLU(1, 256, 6, 2, 1),
            ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
            nn.Conv1d(256, codebook_dimension, 1),
        )
    )


def get_decoder(codebook_dimension):
    return ae.nn.Lightning_Model(
        nn.Sequential(
            nn.Conv1d(codebook_dimension, 256, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
            ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 0),
            nn.Conv1d(256, 1, 1),
        )
    )


def get_vq_vae(codebook_size, codebook_dimension):
    return (
        ae.nn.Lightning_VQ_VAE_codebook_loss(codebook_dimension, codebook_size, True),
        "VQ_loss",
        0.25,
    )


def get_vq_vae_ema(codebook_size, codebook_dimension):
    return (
        ae.nn.Lightning_VQ_VAE_ema(codebook_dimension, codebook_size, 0.99, True),
        "VQ_ema",
        0.25,
    )


def get_ng_vae(codebook_size, codebook_dimension):
    return (
        ae.nn.Lightning_GVQ_VAE_codebook_loss(
            codebook_dimension, codebook_size, 2, None, "linear", True
        ),
        "GAS",
        0.25,
    )


def get_loss():
    return ae.train.MultiscaleSpectralLossL1()


def main(part, num_parts, data):
    codebook_sizes = [256, 512, 1024]  # 3
    codebook_dimensions = [1, 2, 4, 8, 16, 32, 64, 128]  # 8
    vq_models = [get_ng_vae, get_vq_vae, get_vq_vae_ema]  # 3
    # 3 * 8 * 3 = 72

    dataset = ae.data.Speechcommands(data)

    configurations = list(
        itertools.product(codebook_sizes, codebook_dimensions, vq_models)
    )

    part_size = len(configurations) // num_parts
    start = part_size * part
    end = part_size * (part + 1) if part < num_parts - 1 else len(configurations)

    for codebook_size, codebook_dimension, vq_model in configurations[start:end]:

        encoder = get_encoder(codebook_dimension)
        decoder = get_decoder(codebook_dimension)
        vq_layer, name, beta = vq_model(codebook_size, codebook_dimension)

        model = ae.train.QuantizedAutoencoder(
            encoder, decoder, vq_layer, get_loss(), beta
        )

        name = "_".join([name, "s", str(codebook_size), "d", str(codebook_dimension)])

        logger = pl.loggers.TensorBoardLogger(
            "./_logs2/ng_vs_vq/Speechcommands", name, default_hp_metric=False
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/" + model.hparams["reconstruction_loss"],
            dirpath="./_checkpoints2/Speechcommand",
            filename="sample-mnist-{epoch:02d}",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            gpus=1,
            callbacks=[checkpoint_callback],
            logger=logger,
            max_epochs=20,
            enable_checkpointing=True,
        )

        trainer.fit(model, dataset)

        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"]
        )

        trainer.test(model, dataset)


if __name__ == "__main__":
    main(args.part, args.num_parts, args.data_dir)
