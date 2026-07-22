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

from fv.fovea import FoveaDims, build_view


class FoveatedWindowDataset(Dataset):
    def __init__(self, arrays: dict, dims: FoveaDims, split: int,
                 pool_mode: str = "avg", pad_mode: str = "edge"):
        mask = arrays["split"] == split
        self.y = arrays["y"][mask]
        self.sample_idx = arrays["sample_idx"][mask]
        self.window_xy = arrays["window_xy"][mask]
        self.images = arrays["images"]            # (S, H, W) uint8, stays uint8 in RAM
        self.dims = dims
        self.pool_mode = pool_mode
        self.pad_mode = pad_mode
        # sample_idx does NOT index images: images_sample_idx maps rows to A indexes
        lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
        self.image_row = np.asarray([lookup[int(s)] for s in self.sample_idx],
                                    dtype=np.int32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        img = self.images[self.image_row[i]]
        wx0, wy0 = int(self.window_xy[i, 0]), int(self.window_xy[i, 1])
        view, _cov = build_view(img, wx0, wy0, self.dims,
                                pool_mode=self.pool_mode, pad_mode=self.pad_mode)
        x = torch.from_numpy(view).unsqueeze(0)          # (1, N, N)
        y = torch.from_numpy(self.y[i].copy())           # (4, 3)
        return x, y
