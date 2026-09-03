"""The probe itself: Conv(1->K) + ReLU -> z -> ConvTranspose(K->1), no bias.

WHAT MUST HOLD IF THIS IS TOUCHED
---------------------------------
1. **The decoder is linear, single-layer and bias-free, and its atoms are
   renormalised to L2 = 1 after EVERY optimiser step.** Without the
   renormalisation the model gets a degenerate escape for free: scale the
   encoder by 0.01 and the decoder by 100 and you get the SAME reconstruction
   with a hundred times less penalty, i.e. it would learn to make `z` SMALL
   instead of SPARSE, and the sweep over lambda would measure nothing. The
   encoder is deliberately left free: it is the side that has to move.

2. **Stride 1 and padding k//2 on both sides**: the resolution is preserved.
   The brief does not want a smaller image, it wants a more generic one at the
   same size.

3. **The encoder replicates the border** (`padding_mode='replicate'`, the same
   as production's `pad_mode: edge`). The decoder CANNOT: PyTorch's
   `nn.ConvTranspose2d` only accepts `padding_mode='zeros'` -- its `__init__`
   raises otherwise (checked with torch 2.14). So the reconstruction of the
   outer ring of k//2 pixels sees zeros where the encoder saw a replicated
   border. With k=9 that is 4 of every 10 pixels per side, so it is NOT a
   detail: that is why the error is also reported over the interior
   (`err_rec_int`), which is the clean figure. A test fails if a future torch
   accepts it, so that the decision gets revisited instead of just sitting here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Probe(nn.Module):
    """A one-layer-per-side convolutional autoencoder. The model *is* the kernels.

    Nothing sits between encoder and decoder: no batchnorm, no pooling. There is
    nothing behind `z` that could rescue a bad code, which is exactly what does
    NOT happen in `fov16-optimo-mask`, where a 153,660-parameter head sits
    downstream and can extract corners out of almost any projection.
    """

    def __init__(self, channels: int, k: int) -> None:
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"k must be odd so that padding k//2 preserves size (got {k})")
        self.K, self.k = channels, k
        self.enc = nn.Conv2d(1, channels, k, stride=1, padding=k // 2,
                             padding_mode="replicate", bias=True)
        self.dec = nn.ConvTranspose2d(channels, 1, k, stride=1, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.enc(x))
        return self.dec(z), z

    @torch.no_grad()
    def renormalize(self) -> None:
        """Every decoder atom back to L2 norm = 1. Called after EVERY step."""
        w = self.dec.weight                      # (K, 1, k, k)
        n = w.flatten(1).norm(dim=1).clamp_min(1e-8)
        w.div_(n.view(-1, 1, 1, 1))

    # -- the two kernel sets, as (K, k*k) matrices -------------------------
    #
    # No flip is applied between them, and that is a decision with a reason:
    # `conv_transpose2d(w)` is the exact adjoint of `conv2d(w)` (it is how
    # PyTorch backpropagates a convolution to its input), so "tied weights"
    # here means `dec.weight == enc.weight` element-wise, same orientation.
    # Comparing kernel i of one against kernel i of the other is therefore
    # comparing the same object, which is what metric 8 of the brief asks.

    def encoder_kernels(self) -> torch.Tensor:
        return self.enc.weight.detach().flatten(1)

    def decoder_kernels(self) -> torch.Tensor:
        return self.dec.weight.detach().flatten(1)
