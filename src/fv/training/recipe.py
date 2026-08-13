"""D — the recipe: hyperparameters that define the result.

The catalogue is inherited from the sibling with its measured traps set on
purpose: momentum explicit (SGD at 0 loses every comparison), smooth_l1_beta
explicit (the torch default 1.0 with coords in [0,1] silently makes the
position loss pure MSE), scheduler explicit. device/num_workers are NOT here
(X, contract (10)); batch_size IS here (it changes the weights).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fv import settings
from fv.ioutils import strip_envelope, with_envelope
from fv.metrics import MONITORS


@dataclass
class Recipe:
    lr: float = 1e-3
    optimizer: str = "adam"          # adam | adamw | sgd
    momentum: float = 0.9            # applies to sgd; explicit, never the default 0
    weight_decay: float = 0.0
    batch_size: int = 64             # D, not X: changing it changes the result
    epochs: int = 5
    scheduler: str = "none"          # none | cosine
    patience: int = 0                # early stop epochs without improvement; 0 = off
    lambda_pos: float = 1.0          # weight of the position term vs existence
    pos_weight: float = 1.0          # positive-class weight in the BCE
    smooth_l1_beta: float = 0.08     # quadratic->linear threshold; default 1.0 is a trap
    monitor: str = "val_loss"        # explicit, never hardcoded
    seed: int = 1                    # the REPLICA axis, not a hyperparameter to optimise

    def as_dict(self) -> dict:
        return asdict(self)


class RecipeStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def _check_monitor(cfg: dict, where: str) -> None:
    """A monitor names a val metric WITH its direction. 'f1' (the objective) is
    not one: monitor_key finds the value but nothing knows higher is better, so
    best.pt would keep the worst epoch and no one would be told. Refused at the
    gate, in words, instead of a plausible number half an hour later (R4)."""
    m = cfg.get("monitor")
    if m is not None and m not in MONITORS:
        raise RecipeStoreError(
            "unknown_monitor", f"{where}: '{m}' no es un monitor",
            f"usa uno de {sorted(MONITORS)} — el monitor nombra la metrica de "
            f"val ('val_f1'), no el objetivo ('f1')")


class RecipeStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.recipes_root()

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for f in sorted(self.root.glob("*.yaml")):
            # the envelope stays in the file: what is listed is the recipe, so a
            # screen can send a listed row straight back to save()
            cfg = strip_envelope(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
            cfg["name"] = f.stem
            out.append(cfg)
        return out

    def get(self, name: str) -> Recipe:
        f = self.root / f"{name}.yaml"
        if not f.exists():
            known = ", ".join(x.stem for x in self.root.glob("*.yaml")) \
                if self.root.exists() else ""
            raise RecipeStoreError("recipe_not_found",
                                   f"no existe la receta '{name}'",
                                   f"las recetas disponibles son: {known or '(ninguna)'}")
        cfg = strip_envelope(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
        bad = set(cfg) - set(Recipe().as_dict())
        if "device" in bad or "num_workers" in bad:
            raise RecipeStoreError(
                "execution_inside_recipe",
                f"la receta '{name}' lleva dentro campos de ejecucion: {sorted(bad)}",
                "device y num_workers son X (contrato 10): van aparte, no en la receta")
        if bad:
            raise RecipeStoreError("unknown_recipe_fields",
                                   f"la receta '{name}' trae campos desconocidos: {sorted(bad)}",
                                   f"los validos son: {sorted(Recipe().as_dict())}")
        _check_monitor(cfg, f"la receta '{name}'")
        return Recipe(**cfg)

    def save(self, name: str, cfg: dict, overwrite: bool = False) -> None:
        for banned in ("device", "num_workers"):
            if banned in cfg:
                raise RecipeStoreError(
                    "execution_inside_recipe",
                    f"'{banned}' no es un campo de receta",
                    "device y num_workers son X (contrato 10): van en Entrenar, no aqui")
        cfg = strip_envelope(cfg)   # the envelope is not an unknown field
        bad = set(cfg) - set(Recipe().as_dict())
        if bad:
            raise RecipeStoreError("unknown_recipe_fields",
                                   f"campos desconocidos: {sorted(bad)}",
                                   f"los validos son: {sorted(Recipe().as_dict())}")
        _check_monitor(cfg, f"la receta '{name}'")
        self.root.mkdir(parents=True, exist_ok=True)
        f = self.root / f"{name}.yaml"
        if f.exists() and not overwrite:
            raise RecipeStoreError("recipe_exists",
                                   f"ya existe una receta llamada '{name}'",
                                   "elige otro nombre, o edita esa")
        f.write_text(yaml.safe_dump(with_envelope(cfg), sort_keys=False),
                     encoding="utf-8")

    def delete(self, name: str) -> None:
        f = self.root / f"{name}.yaml"
        if not f.exists():
            raise RecipeStoreError("recipe_not_found",
                                   f"no existe la receta '{name}'", "nada que borrar")
        f.unlink()
