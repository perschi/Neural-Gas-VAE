import torch
import torch.nn as nn
import torch.nn.functional as F

import einops as ein


class VQ_VAE_codebook_loss(nn.Module):
    def __init__(self, in_features, codebook_size, research_mode=False):
        super(VQ_VAE_codebook_loss, self).__init__()

        self.codebook = nn.Parameter(
            torch.randn(codebook_size, in_features) / in_features
        )
        self.research_mode = research_mode

    def distance(self, x):
        """Calculate the distance between the flat input and the codebook

        Args:
          x : Flattened inputs of shape [N C P]

        Returns:
          distance: distance matrix [N S P]
        """
        return ((x[:, None] - self.codebook[None, ..., None]) ** 2).sum(dim=2)

    def forward(self, x):
        """Calculates the forward pass including commitment and codebook loss

        Args:
          x : Quantized tensor of shape [N C ...]

        Returns:
          dictionary containing:
             loss_codebook : loss for the codebook
             loss_commitment : loss to commit encoder to quantization
             codebook_indices : the assigned indices to the present vectors
             output : the quantized representation of x
        """

        output = dict()
        x_flat = x.reshape([*x.shape[:2], -1])

        encoding_indices = self.distance(x_flat).min(dim=1)[1]
        x_quantized = ein.rearrange(
            self.codebook[encoding_indices], "N P C -> N C P"
        ).reshape(x.shape)

        if self.training or self.research_mode:
            output["loss_codebook"] = F.mse_loss(x.detach(), x_quantized)
            output["loss_commitment"] = F.mse_loss(x, x_quantized.detach())

        output["codebook_indices"] = encoding_indices.reshape([-1, *x.shape[2:]])
        output["output"] = x + (x_quantized - x).detach()
        return output


class VQ_VAE_ema(nn.Module):
    def __init__(self, in_features, codebook_size, decay=0.99, research_mode=False):
        super(VQ_VAE_ema, self).__init__()
        self.research_mode = research_mode

        self.register_buffer(
            "codebook", torch.randn(codebook_size, in_features) / in_features
        )
        self.register_buffer("_ema_w", self.codebook)
        self.register_buffer("_cluster_size", torch.zeros(codebook_size))

        self.decay = decay
        self.steps = 1
        self._epsilon = 1e-7

    def distance(self, x):
        """Calculate the distance between the flat input and the codebook

        Args:
          x : Flattened inputs of shape [N C P]

        Returns:
          distance: distance matrix [N S P]
        """
        return ((x[:, None] - self.codebook[None, ..., None]) ** 2).sum(dim=2)

    def ema_update(self, parameter, value):
        """Updates the given buffer and returns bias corrected version

        Args:
          x : buffer input
          value : update value of same shape as x

        Returns:
          tensor : zero bias corrected version of the current moving average
        """
        parameter.data = self.decay * parameter + (1 - self.decay) * value
        return parameter / (1 - self.decay ** self.steps)

    def forward(self, x):
        """Calculates the forward pass including commitment loss

        Args:
          x : Quantized tensor of shape [N C ...]

        Returns:
          dictionary containing:
             loss_codebook : loss for the codebook
             loss_commitment : loss to commit encoder to quantization
             codebook_indices : the assigned indices to the present vectors
             output : the quantized representation of x
        """
        output = dict()

        x_flat = x.reshape([*x.shape[:2], -1])

        encoding_indices = self.distance(x_flat).min(dim=1)[1]
        x_quantized = ein.rearrange(
            self.codebook[encoding_indices], "N P C -> N C P"
        ).reshape(x.shape)

        if self.training or self.research_mode:
            output["loss_commitment"] = F.mse_loss(x, x_quantized.detach())

            if self.training:
                with torch.no_grad():
                    # N P C
                    one_hot_indices = ein.rearrange(
                        F.one_hot(encoding_indices, self.codebook.shape[0]),
                        "N P C -> (N P) C",
                    ).type(torch.float32)

                    cluster_size = self.ema_update(
                        self._cluster_size, one_hot_indices.sum(dim=0)
                    )

                    # unnormalized representation
                    dw = torch.einsum(
                        "nj,ni->ij",
                        ein.rearrange(x_flat, "N C P -> (N P) C"),
                        one_hot_indices,
                    )

                    ema_w = self.ema_update(self._ema_w, dw)

                    n = cluster_size.sum()

                    updated_cluster_size = (
                        (cluster_size + self._epsilon)
                        / (n + self.codebook.shape[0] * self._epsilon)
                        * n
                    )

                    self.codebook = ema_w / ein.rearrange(
                        updated_cluster_size, "C -> C 1"
                    )

                    self.steps += 1

        output["codebook_indices"] = encoding_indices.reshape([-1, *x.shape[2:]])
        output["output"] = x + (x_quantized - x).detach()
        return output
