if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Start the grid training on CIFAR")
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
        "--data_dir", nargs="?", type=str, required=True, help='path to the data set (download if not available)'
    )


    args = parser.parse_args()
    
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

def main(part, num_parts, data):
    codebook_sizes = [256, 512] # 2
    codebook_dimensions = [2, 16, 32, 64] # 4
    initial_influences = [0.25, 0.5, 1, 2, 4, 8] # 6
    decay_strategies = ['constant','linear', 'cos'] # 3
    betas = [0.25, 0.5, 1] # 3
    # 2 * 4 * 6 * 3 * 3= 144_c
    
    dataset = ae.data.CIFAR10(data, 64)

    configurations = list(itertools.product(codebook_sizes,codebook_dimensions, initial_influences,decay_strategies, betas))

    part_size = len(configurations) // num_parts
    start = part_size * part
    end = part_size * (part + 1) if part < num_parts -1 else len(configurations)

    
    for codebook_size, codebook_dimension, initial_influence, decay_strategy, beta in configurations[start:end]:

        encoder = get_encoder(codebook_dimension)
        decoder = get_decoder(codebook_dimension)
        vq_layer = ae.nn.Lightning_GVQ_VAE_codebook_loss(codebook_dimension, codebook_size,
                                                         initial_influence,
                                                         None,
                                                         decay_strategy,
                                                         research_mode=True)


        model = ae.train.QuantizedAutoencoder(encoder, decoder, vq_layer, nn.MSELoss(), beta)

        name = "_".join(['GAS',
                    's', str(codebook_size),
                    'd', str(codebook_dimension),
                    'i', str(initial_influence),
                         str(decay_strategy)])
    
        logger = pl.loggers.TensorBoardLogger('./_logs/neural_gas_parameters/CIFAR',
                                              name,
                                              default_hp_metric=False)

        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=100, enable_checkpointing=False)

    
        trainer.fit(model, dataset)


if __name__ == '__main__':
    main(args.part, args.num_parts, args.data_dir)

