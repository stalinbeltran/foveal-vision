import sys, json; sys.path.insert(0,'src')
sys.path.insert(0,'experimentos/2026-09-03-cnn-plana-4k7/nn')
import numpy as np, torch, torch.nn.functional as F, yaml
from fv.models.builder import build_model, full_config
from fv.training.registry import RunStore
from aplicar_kernels import entradas, EXP

vent = json.loads((EXP/"evaluacion"/"set-visualizacion.json").read_text())["ventanas"]
x, e, vistas = entradas(vent)              # (10, 2, 20, 20)

ck = RunStore().path("fov16-mask-p20")/"best.pt"
cfg = json.loads((RunStore().path("fov16-mask-p20")/"config.json").read_text())
net = full_config(cfg["network"] if "network" in cfg else cfg)
m = build_model(net)
st = torch.load(ck, map_location="cpu", weights_only=False)
m.load_state_dict(st["model"] if "model" in st else st["state_dict"]); m.eval()
print("red:", net["regions"], "· capas", net["n_layers"], "· k", net["k_center"],
      "· params", sum(p.numel() for p in m.parameters()))

def rama(convs, xin, modo):
    """Igual que _branch_forward pero con el relleno elegible."""
    h = xin
    for i, cv in enumerate(convs):
        pad = cv.padding[0]
        if modo == "replicate":
            h = F.conv2d(F.pad(h, (pad,)*4, mode="replicate"), cv.weight, cv.bias, cv.stride)
        else:
            h = cv(h)
        if i < len(convs) - 1:
            h = F.relu(h)
    return h

cm = m.center_mask; pm = m.periph_mask
xc = x[:, :1] * cm                       # el centro ve 1 canal, enmascarado
xp = x * pm                              # la periferia ve los 2
print()
for nom, convs, xin in (("centro", m.center_convs, xc), ("periferia", m.periph_convs, xp)):
    a = rama(convs, xin, "zeros").detach().numpy()
    b = rama(convs, xin, "replicate").detach().numpy()
    d = np.abs(a - b)
    esc = np.abs(a).mean()
    # celdas donde el relleno cambia el resultado mas de un 1% de la escala tipica
    afect = (d.mean(axis=(0,1)) > 0.01*esc)
    print(f"  {nom:>10}: |salida| media {esc:.4f} · |cambio| medio {d.mean():.4f} "
          f"({d.mean()/esc*100:4.1f} %) · celdas afectadas {afect.sum()}/{afect.size} "
          f"({afect.sum()/afect.size*100:.0f} %)")
    # cuanto de gordo es el cambio EN las celdas afectadas
    if afect.any():
        print(f"  {'':>10}  en las afectadas, |cambio| = {d.mean(axis=(0,1))[afect].mean()/esc*100:.0f} % de la escala")
