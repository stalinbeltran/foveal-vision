"""Contract (4): the checkpoint describes itself — load_model rebuilds the net
(foveated geometry included) without any YAML or dataset."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from fv.models.builder import (NETWORK_DEFAULTS, FoveatedRegionalNN,
                              build_model)


class CheckpointError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def load_model(ckpt_path: Path, device: str = "cpu") -> FoveatedRegionalNN:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]["model"]
    except Exception as e:
        # Un fichero que NO es un checkpoint. Antes esto no podia pasar --los
        # `.pt` los escribia el propio bucle de entrenamiento-- y desde que hay
        # un endpoint que recibe bytes (`PUT /inference/staging/...`) si puede:
        # una subida cortada, un fichero equivocado, cualquier cosa. Sin esto es
        # un 500 opaco en la pantalla, que es el peor sitio para enterarse.
        raise CheckpointError(
            "checkpoint_ilegible",
            f"{ckpt_path.name} no es un checkpoint de este proyecto: {e}",
            "vuelve a subirlo (una subida cortada deja bytes validos pero "
            "incompletos), o borra la antesala con DELETE "
            "/inference/staging/<run>") from e
    # ⚠ ANTES de construir: ¿declara este checkpoint campos que este proceso no
    # conoce? Entonces no es el checkpoint el que se quedo atras -- es el CODIGO
    # QUE CORRE AQUI, y las dos averias tienen el mismo sintoma con arreglos
    # opuestos: una pide reiniciar (gratis), la otra reentrenar (dinero).
    #
    # Paso el 2026-09-01: la web app llevaba corriendo desde antes de que
    # existiera `mask_channel`, construia la rama periferica con un canal, y los
    # pesos traian dos. El mensaje generico decia "reentrena el run" -- o sea,
    # gastar en Vast para arreglar un modelo que estaba perfecto.
    #
    # Y se mira aqui y no en el `except` porque un campo nuevo que NO cambie
    # ninguna forma se cargaria sin protestar y la red haria otra cosa que la
    # que su config declara: silencioso, que es peor que el fallo ruidoso.
    desconocidos = sorted(set(cfg) - set(NETWORK_DEFAULTS))
    if desconocidos:
        raise CheckpointError(
            "checkpoint_de_codigo_mas_nuevo",
            f"{ckpt_path.name} declara campos que este proceso no conoce: "
            f"{', '.join(desconocidos)}. El checkpoint es mas nuevo que el codigo "
            f"que esta corriendo",
            "reinicia el servicio para que cargue el codigo actual "
            "(sudo systemctl restart foveal-vision-web) o, en desarrollo, el "
            "proceso de fv-api. NO reentrenes: los pesos estan bien")
    model = build_model(cfg)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        # a checkpoint from a previous builder (e.g. the fixed two-layer conv1/
        # conv2 before the parametric builder) no longer fits — no weight-compat
        # code is written on purpose (D-C2 §13). Fail with the reason, never a 500.
        raise CheckpointError(
            "checkpoint_incompatible",
            "este checkpoint es de un builder anterior y sus pesos ya no encajan "
            "en la red parametrica",
            "reentrena el run (fv-train / un recorrido): no se migra state_dict "
            "(barrido-por-ejes.md §13). ⚠ Si el run es RECIENTE, mira antes si "
            "el proceso lleva vivo desde antes que el: reiniciarlo es gratis y "
            "reentrenar cuesta") from e
    model.to(device)
    model.eval()
    return model


class ModelCache:
    """Keyed by (path, device, mtime): a live run rewrites best.pt every epoch
    it improves — without the mtime you would serve the first epoch forever."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, ckpt_path: Path, device: str = "cpu") -> FoveatedRegionalNN:
        p = Path(ckpt_path)
        key = (str(p), device, p.stat().st_mtime_ns)
        if key not in self._cache:
            self._cache.clear()  # keep at most one: models are MBs
            self._cache[key] = load_model(p, device)
        return self._cache[key]


MODEL_CACHE = ModelCache()
