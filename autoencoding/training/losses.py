import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Extracted from the package  https://github.com/asteroid-team/asteroid following the

MIT License

Copyright (c) 2019 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
EPS = 1e-8


def check_complex(tensor, dim=-2):
    """Assert tensor in complex-like in a given dimension.

    Args:
        tensor (torch.Tensor): tensor to be checked.
        dim(int): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Raises:
        AssertionError if dimension is not even in the specified dimension

    """
    if tensor.shape[dim] % 2 != 0:
        raise AssertionError(
            "Could not equally chunk the tensor (shape {}) "
            "along the given dimension ({}). Dim axis is "
            "probably wrong"
        )


def take_mag(x, dim=-2):
    """Takes the magnitude of a complex tensor.

    The operands is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`: The magnitude of x.
    """
    check_complex(x, dim=dim)
    power = torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)
    power = power + EPS
    return power.pow(0.5)


class Filterbank(nn.Module):
    """Base Filterbank class.
    Each subclass has to implement a `filters` property.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the conv or transposed conv. (Hop size).
            If None (default), set to ``kernel_size // 2``.

    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(self, n_filters, kernel_size, stride=None):
        super(Filterbank, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        # If not specified otherwise in the filterbank's init, output
        # number of features is equal to number of required filters.
        self.n_feats_out = n_filters

    @property
    def filters(self):
        """Abstract method for filters."""
        raise NotImplementedError

    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class."""
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
        }
        return config


class _EncDec(nn.Module):
    """Base private class for Encoder and Decoder.

    Common parameters and methods.

    Args:
        filterbank (:class:`Filterbank`): Filterbank instance. The filterbank
            to use as an encoder or a decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.

    Attributes:
        filterbank (:class:`Filterbank`)
        stride (int)
        is_pinv (bool)
    """

    def __init__(self, filterbank, is_pinv=False):
        super(_EncDec, self).__init__()
        self.filterbank = filterbank
        self.stride = self.filterbank.stride
        self.is_pinv = is_pinv

    @property
    def filters(self):
        return self.filterbank.filters

    def compute_filter_pinv(self, filters):
        """Computes pseudo inverse filterbank of given filters."""
        scale = self.filterbank.stride / self.filterbank.kernel_size
        shape = filters.shape
        ifilt = torch.pinverse(filters.squeeze()).transpose(-1, -2).view(shape)
        # Compensate for the overlap-add.
        return ifilt * scale

    def get_filters(self):
        """Returns filters or pinv filters depending on `is_pinv` attribute"""
        if self.is_pinv:
            return self.compute_filter_pinv(self.filters)
        else:
            return self.filters

    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class."""
        config = {"is_pinv": self.is_pinv}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(_EncDec):
    """Encoder class.

    Add encoding methods to Filterbank classes.
    Not intended to be subclassed.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use
            as an encoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        as_conv1d (bool): Whether to behave like nn.Conv1d.
            If True (default), forwarding input with shape (batch, 1, time)
            will output a tensor of shape (batch, freq, conv_time).
            If False, will output a tensor of shape (batch, 1, freq, conv_time).
        padding (int): Zero-padding added to both sides of the input.

    """

    def __init__(self, filterbank, is_pinv=False, as_conv1d=True, padding=0):
        super(Encoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.padding = padding

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        """Returns an :class:`~.Encoder`, pseudo inverse of a
        :class:`~.Filterbank` or :class:`~.Decoder`."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True, **kwargs)
        elif isinstance(filterbank, Decoder):
            return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def forward(self, waveform):
        """Convolve input waveform with the filters from a filterbank.
        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.
        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.

        Shapes:
            >>> (time, ) --> (freq, conv_time)
            >>> (batch, time) --> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) --> (batch, freq, conv_time)
            >>>     (batch, chan, time) --> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) --> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) --> (batch, any, dim, freq, conv_time)
        """
        filters = self.get_filters()
        if waveform.ndim == 1:
            # Assumes 1D input with shape (time,)
            # Output will be (freq, conv_time)
            return F.conv1d(
                waveform[None, None], filters, stride=self.stride, padding=self.padding
            ).squeeze()
        elif waveform.ndim == 2:
            # Assume 2D input with shape (batch or channels, time)
            # Output will be (batch or channels, freq, conv_time)
            warnings.warn(
                "Input tensor was 2D. Applying the corresponding "
                "Decoder to the current output will result in a 3D "
                "tensor. This behaviours was introduced to match "
                "Conv1D and ConvTranspose1D, please use 3D inputs "
                "to avoid it. For example, this can be done with "
                "input_tensor.unsqueeze(1)."
            )
            return F.conv1d(
                waveform.unsqueeze(1), filters, stride=self.stride, padding=self.padding
            )
        elif waveform.ndim == 3:
            batch, channels, time_len = waveform.shape
            if channels == 1 and self.as_conv1d:
                # That's the common single channel case (batch, 1, time)
                # Output will be (batch, freq, stft_time), behaves as Conv1D
                return F.conv1d(
                    waveform, filters, stride=self.stride, padding=self.padding
                )
            else:
                # Return batched convolution, input is (batch, 3, time),
                # output will be (batch, 3, freq, conv_time).
                # Useful for multichannel transforms
                # If as_conv1d is false, (batch, 1, time) will output
                # (batch, 1, freq, conv_time), useful for consistency.
                return self.batch_1d_conv(waveform, filters)
        else:  # waveform.ndim > 3
            # This is to compute "multi"multichannel convolution.
            # Input can be (*, time), output will be (*, freq, conv_time)
            return self.batch_1d_conv(waveform, filters)

    def batch_1d_conv(self, inp, filters):
        # Here we perform multichannel / multi-source convolution. Ou
        # Output should be (batch, channels, freq, conv_time)
        batched_conv = F.conv1d(
            inp.view(-1, 1, inp.shape[-1]),
            filters,
            stride=self.stride,
            padding=self.padding,
        )
        output_shape = inp.shape[:-1] + batched_conv.shape[-2:]
        return batched_conv.view(output_shape)


class STFTFB(Filterbank):
    """STFT filterbank.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.

    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(self, n_filters, kernel_size, stride=None, window=None, **kwargs):
        super(STFTFB, self).__init__(n_filters, kernel_size, stride=stride)
        assert n_filters >= kernel_size
        self.cutoff = int(n_filters / 2 + 1)
        self.n_feats_out = 2 * self.cutoff

        if window is None:
            self.window = np.hanning(kernel_size + 1)[:-1] ** 0.5
        else:
            ws = window.size
            if not (ws == kernel_size):
                raise AssertionError(
                    "Expected window of size {}."
                    "Received window of size {} instead."
                    "".format(kernel_size, ws)
                )
            self.window = window
        # Create and normalize DFT filters (can be overcomplete)
        filters = np.fft.fft(np.eye(n_filters))
        filters /= 0.5 * np.sqrt(kernel_size * n_filters / self.stride)

        # Keep only the windowed centered part to save computation.
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        indexes = list(range(lpad, n_filters - rpad))
        filters = np.vstack(
            [
                np.real(filters[: self.cutoff, indexes]),
                np.imag(filters[: self.cutoff, indexes]),
            ]
        )

        filters[0, :] /= np.sqrt(2)
        filters[n_filters // 2, :] /= np.sqrt(2)
        filters = torch.from_numpy(filters * self.window).unsqueeze(1).float()
        self.register_buffer("_filters", filters)

    @property
    def filters(self):
        return self._filters


class MultiscaleSpectralLossL1(nn.Module):
    """Measure multi-scale spectral loss as described in [1]

    Args:
        n_filters (list): list containing the number of filter desired for
            each STFT
        windows_size (list): list containing the size of the window desired for
            each STFT
        hops_size (list): list containing the size of the hop desired for
            each STFT

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of training targets.
        alpha (float) : Weighting factor for the log term

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples:
        >>> import torch
        >>> targets = torch.randn(10, 32000)
        >>> est_targets = torch.randn(10, 32000)
        >>> # Using it by itself on a pair of source/estimate
        >>> loss_func = SingleSrcMultiScaleSpectral()
        >>> loss = loss_func(est_targets, targets)

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # Using it with PITLossWrapper with sets of source/estimates
        >>> loss_func = PITLossWrapper(SingleSrcMultiScaleSpectral(),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References:
        [1] Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and
        Adam Roberts DDSP: Differentiable Digital Signal Processing
        International Conference on Learning Representations ICLR 2020 $
    """

    def __init__(self, n_filters=None, windows_size=None, hops_size=None, alpha=1.0):
        super(MultiscaleSpectralLossL1, self).__init__()

        if windows_size is None:
            windows_size = [2048, 1024, 512, 256, 128, 64, 32]
        if n_filters is None:
            n_filters = [2048, 1024, 512, 256, 128, 64, 32]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128, 64, 32, 16]

        self.windows_size = windows_size
        self.n_filters = n_filters
        self.hops_size = hops_size
        self.alpha = alpha

        self.encoders = nn.ModuleList(
            Encoder(STFTFB(n_filters[i], windows_size[i], hops_size[i]))
            for i in range(len(self.n_filters))
        )

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)

        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)
        return loss.mean()

    def compute_spectral_loss(self, encoder, est_target, target):
        batch_size = est_target.shape[0]
        spect_est_target = take_mag(encoder(est_target)).view(batch_size, -1)
        spect_target = take_mag(encoder(target)).view(batch_size, -1)
        linear_loss = self.norm1(spect_est_target - spect_target)
        log_loss = self.norm1(
            torch.log(spect_est_target + EPS) - torch.log(spect_target + EPS)
        )
        return linear_loss + self.alpha * log_loss

    @staticmethod
    def norm1(a):
        return torch.norm(a, p=1, dim=1)
