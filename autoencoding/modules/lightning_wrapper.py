import pytorch_lightning as pl

from .vqvae import VQ_VAE_codebook_loss, VQ_VAE_ema
from .gvqvae import GVQ_VAE_codebook_loss


class Lightning_VQ_VAE_codebook_loss(pl.LightningModule):
    def __init__(self, in_features, codebook_size, research_mode=False):
        super().__init__()
        self.save_hyperparameters()
        self.vq = VQ_VAE_codebook_loss(in_features, codebook_size, research_mode)

    def forward(self, x):
        return self.vq(x)


class Lightning_VQ_VAE_ema(pl.LightningModule):
    def __init__(self, in_features, codebook_size, decay=0.99, research_mode=False):
        super().__init__()
        self.save_hyperparameters()
        self.vq = VQ_VAE_ema(
            in_features, codebook_size, decay, research_mode=research_mode
        )

    def forward(self, x):
        return self.vq(x)


class Lightning_GVQ_VAE_codebook_loss(pl.LightningModule):
    def __init__(
        self,
        in_features,
        codebook_size,
        influence=None,
        lifetime=10,
        influence_decay="None",
        research_mode=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vq = GVQ_VAE_codebook_loss(
            in_features,
            codebook_size,
            influence,
            lifetime,
            influence_decay,
            research_mode=research_mode,
        )

    def forward(self, x):
        return self.vq(x)


class Lightning_Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
