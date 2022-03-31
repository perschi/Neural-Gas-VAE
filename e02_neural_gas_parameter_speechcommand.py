if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Start the grid training on Speech Commands")
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
    return  ae.nn.Lightning_Model(nn.Sequential(ae.nn.Conv1d_CBnReLU(1, 256, 6, 2, 1),
                                                ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
                                                ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
                                                ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
                                                ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
                                                ae.nn.Conv1d_CBnReLU(256, 256, 6, 2, 1),
                                                nn.Conv1d(256, codebook_dimension, 1)
                                                )
                                  )

def get_decoder(codebook_dimension):
    return ae.nn.Lightning_Model(nn.Sequential(   nn.Conv1d(codebook_dimension, 256, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 1),
                                                  ae.nn.ConvTransposed1d_CBnReLU(256, 256, 6, 2, 0),
                                                  nn.Conv1d(256, 1, 1)
                                                  )
                                  )


def get_loss():
    return ae.train.MultiscaleSpectralLossL1()


def main(part, num_parts, data):
    codebook_sizes =  [512] # [256, 512]
    codebook_dimensions = [2, 16, 32] #[2, 16, 32, 64]
    initial_influences = [0.5, 1, 2, 4] # [0.25, 0.5, 1, 2, 4]
    decay_strategies = ['constant','linear', 'cos']
    betas = [0.1, 0.25, 0.5]
    # 2 * 4 * 5 * 3 = 120 -> 10 * 12
    # 1 * 3 * 4 * 3  * 3 = 108
    
    dataset = ae.data.Speechcommands(data)

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


        model = ae.train.QuantizedAutoencoder(encoder, decoder, vq_layer, get_loss(), beta)

        name = "_".join(['GAS',
                    's', str(codebook_size),
                    'd', str(codebook_dimension),
                    'i', str(initial_influence),
                         str(decay_strategy)])
    
        logger = pl.loggers.TensorBoardLogger('./_logs/neural_gas_parameters/Speechcommands',
                                              name,
                                              default_hp_metric=False)

        
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=20, enable_checkpointing=False)

    
        trainer.fit(model, dataset)


if __name__ == '__main__':
    main(args.part, args.num_parts, args.data_dir)




