"""L1 probe: can first-layer kernels learn generic filters when under pressure?

WHY THIS IS A PACKAGE OF ITS OWN, AND WHAT IT MAY IMPORT
--------------------------------------------------------
The probe is a SEPARATE experiment. It does not touch `fv.models.builder`, nor
the configs under `configs/networks/`, so no checkpoint on disk changes meaning
because this package exists.

The isolation rule (from the brief, section 6): this package must not import
`fv.models` nor `fv.fovea`, *except* for the window loader. The one exception
taken is `fv.fovea.build_view` / `fv.fovea.dims_of` in `fv.probe.data`, which
IS that loader: it is the function that turns a stored window into the 20x20
view the production network consumes (contract (5)). Rebuilding the view here
would measure a datum the network never sees, which is the one thing that would
make every number below meaningless.

`fv.models` is NOT imported anywhere in this package. The production geometry
travels in as a plain dict argument, supplied by the caller
(`scripts/sonda_l1.py`), so the single source of truth stays in `builder.py`
without this package depending on it.

WHAT THIS PROBE CANNOT SAY
--------------------------
That a 9x9 kernel trained here is generic does NOT say the production 3x3 could
be. Those are two questions -- "does the structure fit?" and "is there
pressure?" -- and the probe moves both at once. That is why k=3 is in the grid
as an ANCHOR: it is the only arm comparable with the 0.688 already measured on
`fov16-mask-p20`.
"""

from fv.probe.data import local_contrast_norm, prepare
from fv.probe.figures import contact_sheet, code_maps
from fv.probe.gabor import fit_gabor_r2, random_baseline_r2
from fv.probe.metrics import classic_basis, final_metrics
from fv.probe.model import L1Probe
from fv.probe.run import GRID_CHANNELS, GRID_KS, GRID_KS_ANCHOR, GRID_LAMBDAS, run_name, train
from fv.probe.table import comparison_table

__all__ = [
    "L1Probe",
    "local_contrast_norm",
    "prepare",
    "classic_basis",
    "final_metrics",
    "fit_gabor_r2",
    "random_baseline_r2",
    "train",
    "run_name",
    "GRID_KS",
    "GRID_KS_ANCHOR",
    "GRID_CHANNELS",
    "GRID_LAMBDAS",
    "contact_sheet",
    "code_maps",
    "comparison_table",
]
