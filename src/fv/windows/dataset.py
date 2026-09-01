"""The torch Dataset: builds the foveated view lazily, per item (contract (5)).

The view comes from THE SAME fv.fovea functions inference uses — the test
asserts the seam, not the function. uint8 images stay in RAM; the composite
view is built per item with reduceat-based pooling (C-speed; the python
double-loop trap measured 48x slower in the sibling is avoided by design).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from fv.fovea import (FoveaDims, build_view, edge_features, input_stack,
                      n_edge_features)


class FoveatedWindowDataset(Dataset):
    def __init__(self, arrays: dict, dims: FoveaDims, split: int,
                 pool_mode: str = "avg", pad_mode: str = "edge",
                 edge_inputs: str = "off", mask_channel: str = "off"):
        mask = arrays["split"] == split
        self.y = arrays["y"][mask]
        self.sample_idx = arrays["sample_idx"][mask]
        self.window_xy = arrays["window_xy"][mask]
        self.images = arrays["images"]            # (S, H, W) uint8, stays uint8 in RAM
        self.dims = dims
        self.pool_mode = pool_mode
        self.pad_mode = pad_mode
        # C's extra head inputs about the IMAGE edge. Read here and not derived
        # from `dims`, because it is a choice of the net and not of the geometry:
        # two nets over the same dataset can disagree about it.
        self.edge_inputs = edge_inputs
        self.n_edge = n_edge_features(edge_inputs)
        # idem: el canal de relleno es decision de la red, no del dataset
        self.mask_channel = mask_channel
        # sample_idx does NOT index images: images_sample_idx maps rows to A indexes
        lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
        self.image_row = np.asarray([lookup[int(s)] for s in self.sample_idx],
                                    dtype=np.int32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        img = self.images[self.image_row[i]]
        wx0, wy0 = int(self.window_xy[i, 0]), int(self.window_xy[i, 1])
        view, cov = build_view(img, wx0, wy0, self.dims,
                               pool_mode=self.pool_mode, pad_mode=self.pad_mode)
        # (C, N, N) -- C es 1, o 2 con el canal de relleno. Lo arma `input_stack`
        # y no este fichero: la inferencia tiene que armarlo IGUAL (contrato (5)),
        # y dos sitios que apilan canales acaban discrepando en el orden.
        x = torch.from_numpy(input_stack(view, cov, self.mask_channel))
        y = torch.from_numpy(self.y[i].copy())           # (4, 3)
        # ALWAYS three items, even with edge_inputs='off' -- then `e` is (0,) and
        # the batch is (B, 0), which concatenates to nothing in the head. One
        # unpacking shape for every caller: a loader that yields 2-tuples
        # sometimes and 3-tuples other times is a `for x, y in loader` that
        # breaks in whichever branch nobody ran.
        e = torch.from_numpy(edge_features(self.images.shape[1:], wx0, wy0,
                                           self.dims, self.edge_inputs))
        return x, e, y
