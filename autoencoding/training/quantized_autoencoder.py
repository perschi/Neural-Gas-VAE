import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl

from .util import class_name, entropy


class QuantizedAutoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: pl.LightningModule,
        decoder: pl.LightningModule,
        vq_layer: pl.LightningModule,
        reconstruction_loss: nn.Module,
        beta=1.0,
    ):
        super(QuantizedAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vq_layer = vq_layer

        self.training_progress = (
            "training_progress" in inspect.getargspec(vq_layer.forward)[0]
        )

        self.loss_fn = reconstruction_loss
        self.beta = beta

        hyper_parameters = {
            "reconstruction_loss": class_name(reconstruction_loss),
            "quantization_name": class_name(vq_layer),
            "beta": self.beta,
        }
        hyper_parameters.update(self.encoder.hparams)
        hyper_parameters.update(self.vq_layer.hparams)
        hyper_parameters.update(self.decoder.hparams)

        self.save_hyperparameters(hyper_parameters)

        self.best_metrics = {"best/" + self.hparams["reconstruction_loss"]: 1e7}

    def forward(self, x):
        encoded = self.encoder(x)

        if self.training_progress:
            quantized_output = self.vq_layer(
                encoded, float(self.trainer.current_epoch) / self.trainer.max_epochs
            )
        else:
            quantized_output = self.vq_layer(encoded)

        reconstruction = self.decoder(quantized_output["output"])
        quantized_output.update({"encoding": encoded, "reconstruction": reconstruction})
        return quantized_output

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(
                self.hparams,
                {
                    "val/" + self.hparams["reconstruction_loss"]: 0,
                    "val/codebook_entropy": 0,
                    "val/loss_commitment": 0,
                    "val/loss_codebook": 0,
                    "test/" + self.hparams["reconstruction_loss"]: 0,
                    "test/codebook_entropy": 0,
                    "test/loss_commitment": 0,
                    "test/loss_codebook": 0,
                    "best/" + self.hparams["reconstruction_loss"]: 0,
                    "best/codebook_entropy": 0,
                    "best/loss_commitment": 0,
                    "best/loss_codebook": 0,
                },
            )

    def training_step(self, batch):
        x = batch[0]

        out = self(x)

        losses = {
            self.hparams["reconstruction_loss"]: self.loss_fn(out["reconstruction"], x),
            "loss_commitment": out["loss_commitment"],
        }

        if "loss_codebook" in out:
            losses["loss_codebook"] = out["loss_codebook"]

        loss = sum(
            map(
                lambda x: x[1] if x[0] != "loss_commitment" else self.beta * x[1],
                losses.items(),
            )
        )

        self.log("loss", loss)

        for k, v in losses.items():
            self.log(k, v)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        out = self(x)

        losses = {
            "reconstruction_loss": (
                self.loss_fn(out["reconstruction"], x) * x.shape[0]
            ).detach(),
            "code_appearance": F.one_hot(
                out["codebook_indices"], self.vq_layer.vq.codebook.shape[0]
            )
            .view(-1, self.vq_layer.vq.codebook.shape[0])
            .sum(dim=0)
            .detach(),
            "loss_commitment": (out["loss_commitment"] * x.shape[0]).detach(),
            "samples": x.shape[0],
        }

        if "loss_codebook" in out:
            losses["loss_codebook"] = (out["loss_codebook"] * x.shape[0]).detach()

        return losses

    def validation_epoch_end(self, outputs):
        total_samples = sum(d["samples"] for d in outputs)

        avg_reconstruction_loss = (
            sum(d["reconstruction_loss"] for d in outputs) / total_samples
        )
        self.log("val/" + self.hparams["reconstruction_loss"], avg_reconstruction_loss)

        avg_loss_commitment = sum(d["loss_commitment"] for d in outputs) / total_samples
        self.log("val/loss_commitment", avg_loss_commitment)

        avg_loss_codebook = 0.0
        if "loss_codebook" in outputs[0]:
            avg_loss_codebook = sum(d["loss_codebook"] for d in outputs) / total_samples
            self.log("val/loss_codebook", avg_loss_codebook)

        codebook_distribution = sum(d["code_appearance"] for d in outputs)
        codebook_distribution = (
            codebook_distribution / codebook_distribution.sum()
        ).sort()[0]

        self.log("val/codebook_entropy", entropy(codebook_distribution))

        if self.logger is not None:
            fig = plt.figure()
            plt.bar(
                np.arange(codebook_distribution.shape[0]),
                codebook_distribution.detach().cpu().numpy(),
            )
            self.logger.experiment.add_figure(
                "codebook_usage", fig, global_step=self.global_step
            )

        if (
            self.best_metrics["best/" + self.hparams["reconstruction_loss"]]
            > avg_reconstruction_loss
        ):
            self.best_metrics[
                "best/" + self.hparams["reconstruction_loss"]
            ] = avg_reconstruction_loss
            self.best_metrics["best/codebook_entropy"] = entropy(codebook_distribution)
            self.best_metrics["best/loss_commitment"] = avg_loss_commitment
            self.best_metrics["best/loss_codebook"] = avg_loss_codebook

            for key, value in self.best_metrics.items():
                self.log(key, value)

    def test_step(self, batch, batch_idx):
        x = batch[0]
        out = self(x)

        losses = {
            "reconstruction_loss": (
                self.loss_fn(out["reconstruction"], x) * x.shape[0]
            ).detach(),
            "code_appearance": F.one_hot(
                out["codebook_indices"], self.vq_layer.vq.codebook.shape[0]
            )
            .view(-1, self.vq_layer.vq.codebook.shape[0])
            .sum(dim=0)
            .detach(),
            "loss_commitment": (out["loss_commitment"] * x.shape[0]).detach(),
            "samples": x.shape[0],
        }

        if "loss_codebook" in out:
            losses["loss_codebook"] = (out["loss_codebook"] * x.shape[0]).detach()

        return losses

    def test_epoch_end(self, outputs):
        total_samples = sum(d["samples"] for d in outputs)

        avg_reconstruction_loss = (
            sum(d["reconstruction_loss"] for d in outputs) / total_samples
        )
        self.log("test/" + self.hparams["reconstruction_loss"], avg_reconstruction_loss)

        avg_loss_commitment = sum(d["loss_commitment"] for d in outputs) / total_samples
        self.log("test/loss_commitment", avg_loss_commitment)

        avg_loss_codebook = 0.0
        if "loss_codebook" in outputs[0]:
            avg_loss_codebook = sum(d["loss_codebook"] for d in outputs) / total_samples
            self.log("test/loss_codebook", avg_loss_codebook)

        codebook_distribution = sum(d["code_appearance"] for d in outputs)
        codebook_distribution = (
            codebook_distribution / codebook_distribution.sum()
        ).sort()[0]

        self.log("test/codebook_entropy", entropy(codebook_distribution))

        if self.logger is not None:
            fig = plt.figure()
            plt.bar(
                np.arange(codebook_distribution.shape[0]),
                codebook_distribution.detach().cpu().numpy(),
            )
            self.logger.experiment.add_figure(
                "test/codebook_usage", fig, global_step=self.global_step
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
