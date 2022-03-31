if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Start the grid training of VQ-VAE and Neural-Gas VAE models on CIFAR"
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

    assert(args.part < args.num_parts)
    
    import os

    os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu)

    
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import autoencoding as ae

import itertools


def get_encoder(codebook_dimension):
    return  ae.nn.Lightning_Model(nn.Sequential(ae.nn.Conv2d_CBnReLU(3, 256, 4, 2, 1),
                                                  ae.nn.Conv2d_CBnReLU(256, 256, 4, 2, 1),
                                                  ae.nn.ResidualBlock2d(256, 256),
                                                  ae.nn.ResidualBlock2d(256, 256),
                                                  nn.Conv2d(256, codebook_dimension, 1)
                                                )
                                  )

def get_decoder(codebook_dimension):
    return ae.nn.Lightning_Model(nn.Sequential(
                                                  nn.Conv2d(codebook_dimension, 256, 1),
                                                  ae.nn.ResidualBlock2d(256, 256),
                                                  ae.nn.ResidualBlock2d(256, 256),
                                                  ae.nn.ConvTransposed2d_CBnReLU(256, 256, 4, 2, 1),
                                                  ae.nn.ConvTransposed2d_CBnReLU(256, 256, 4, 2, 1),
                                                  nn.Conv2d(256, 3, 1)
                                                  )
                                  )

def get_vq_vae_loss(codebook_size, codebook_dimension):
    return ae.nn.Lightning_VQ_VAE_codebook_loss(codebook_dimension, codebook_size, True), "VQ_loss", 0.25

def get_vq_vae_ema(codebook_size, codebook_dimension):
    return ae.nn.Lightning_VQ_VAE_ema(codebook_dimension, codebook_size, 0.99, True), "VQ_ema", 0.25

def get_ng_vae(codebook_size, codebook_dimension):
    return ae.nn.Lightning_GVQ_VAE_codebook_loss(codebook_dimension, codebook_size, 2, None, 'linear', True), "GAS", 0.25


def main(part, num_parts, data):
    codebook_sizes = [128, 256, 512, 1024] # 4
    codebook_dimensions = [1, 2, 4, 8, 16, 32, 64, 128] # 8
    vq_models = [get_ng_vae, get_vq_vae_loss, get_vq_vae_ema] # 3
    
    # 4 * 8 * 3 = 96
    dataset = ae.data.CIFAR10(data, 64)

    configurations = list(itertools.product(codebook_sizes,codebook_dimensions, vq_models))

    part_size = len(configurations) // num_parts
    start = part_size * part
    end = part_size * (part + 1) if part < num_parts -1 else len(configurations)

    for codebook_size, codebook_dimension, vq_model in configurations[start:end]:

        encoder = get_encoder(codebook_dimension)
        decoder = get_decoder(codebook_dimension)
        vq_layer, name, beta  = vq_model(codebook_size, codebook_dimension)

        
        model = ae.train.QuantizedAutoencoder(encoder, decoder, vq_layer, nn.MSELoss(), beta)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/"+model.hparams['reconstruction_loss'],
            dirpath="./_checkpoints/CIFAR",
            filename="cifar-{epoch:02d}",
            save_top_k=1,
            mode="min")

        name = "_".join([name,
                    's', str(codebook_size),
                    'd', str(codebook_dimension)])
    
        logger = pl.loggers.TensorBoardLogger('./_logs/ng_vs_vq/CIFAR',
                                              name,
                                              default_hp_metric=False)

        trainer = pl.Trainer(gpus=1,callbacks=[checkpoint_callback], logger=logger, max_epochs=100, enable_checkpointing=True)

    
        trainer.fit(model, dataset)

         
        model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        
        trainer.test(model, dataset)

if __name__ == '__main__':
    main(args.part, args.num_parts, args.data_dir)

