import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import einops as ein


class GVQ_VAE_codebook_loss(nn.Module):
    def __init__(
        self,
        in_features,
        codebook_size,
        influence=None,
        lifetime=None,
        influence_decay="constant",
        research_mode=False,
    ):
        super(GVQ_VAE_codebook_loss, self).__init__()
        assert influence_decay in ["constant", "linear", "cos"]

        self.influence = influence if influence is not None else 1
        assert self.influence > 0

        self.research_mode = research_mode

        self._eps = torch.finfo(torch.float32).eps

        self._idc = lambda x: self.influence

        if influence_decay == "linear":
            self._idc = lambda x: (1 - x) * self.influence + x * self._eps
        if influence_decay == "cos":
            self._idc = (
                lambda x: self.influence * (math.cos(x * torch.pi) + 1) * 0.5
                + x * self._eps
            )

        self.lifetime = lifetime

        self.codebook = nn.Parameter(
            torch.randn(codebook_size, in_features) / in_features
        )
        self.register_buffer(
            "C", torch.zeros(codebook_size, codebook_size, dtype=torch.int64)
        )
        self.register_buffer(
            "T", torch.zeros(codebook_size, codebook_size, dtype=torch.int64)
        )
        self.register_buffer("range", torch.arange(codebook_size))

    def distance(self, x):
        """Calculate the distance between the flat input and the codebook

        Args:
          x : Flattened inputs of shape [N C P]

        Returns:
          distance: distance matrix [N S P]
        """
        return ((x[:, None] - self.codebook[None, ..., None]) ** 2).sum(dim=2)

    def forward(self, x, training_progress=0.0):
        """Calculates the forward pass including commitment and codebook loss
        Args:
          x : Quantized tensor of shape [N C ...]
          training_progress: progress in the complete trainings_progress default 1
              is used to decay the influence depending on the influence decay
              parameter


        Returns:
          dictionary containing:
             loss_codebook : loss for the codebook
             loss_commitment : loss to commit encoder to quantization
             codebook_indices : the assigned indices to the present vectors
             output : the quantized representation of x
        """
        output = dict()

        x_flat = x.reshape([*x.shape[:2], -1])

        distances = self.distance(x_flat.detach())
        encoding_indices = distances.min(dim=1)[1]
        x_quantized = ein.rearrange(
            self.codebook[encoding_indices], "N P C -> N C P"
        ).reshape(x.shape)

        if self.training or self.research_mode:
            output["loss_commitment"] = F.mse_loss(x, x_quantized.detach())

            d_tilde, ascending_order = torch.sort(distances, dim=1, descending=False)

            influence = self._idc(training_progress)

            output["loss_codebook"] = (
                torch.exp(-self.range[None, :, None] / (influence)) * d_tilde
            ).mean()

            if self.training and self.lifetime is not None:
                with torch.no_grad():
                    pairs = ein.rearrange(ascending_order[:, :2], "N C P -> (N P) C")

                    A = (
                        torch.einsum(
                            "nr,nc->rc",
                            F.one_hot(pairs[:, 0], self.C.shape[0]).type(torch.float32),
                            F.one_hot(pairs[:, 1], self.C.shape[1]).type(torch.float32),
                        )
                        > 0
                    )

                    # everywhere where A is True  set  T to 0
                    Ai = (
                        F.one_hot(pairs[:, 0], self.C.shape[0])
                        .type(torch.float32)
                        .sum(dim=0)
                        > 0
                    ) & ~A
                    # increase T where Ai is true

                    self.T = torch.where(
                        A == 1,
                        torch.zeros_like(self.T, dtype=torch.int64),
                        torch.where(Ai, self.T + 1, A.type(torch.int64)),
                    )

                    self.C = torch.maximum(self.C, A)
                    self.C = torch.where(
                        Ai & (self.T > self.lifetime), torch.zeros_like(self.C), self.C
                    )

        output["codebook_indices"] = encoding_indices.reshape([-1, *x.shape[2:]])
        output["output"] = x + (x_quantized - x).detach()

        return output
