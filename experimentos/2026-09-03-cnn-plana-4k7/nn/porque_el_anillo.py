import sys, json; sys.path.insert(0,'src')
sys.path.insert(0,'experimentos/2026-09-03-cnn-plana-4k7/nn')
import numpy as np, torch, torch.nn.functional as F
from aplicar_kernels import entradas, modelo, EXP

vent = json.loads((EXP/"evaluacion"/"set-visualizacion.json").read_text())["ventanas"]
x, e, vistas = entradas(vent)
m, etq = modelo("plana-4k7-s1")
conv = m.center_convs[0]; b = conv.kernel_size[0]//2; c = slice(b,-b)

print("=== 1. ¿tiene esta red mascaras de region? ===")
buf = list(m.named_buffers())
print("   buffers del modelo:", [n for n,_ in buf] or "NINGUNO")
print("   regions:", "single" if getattr(m,'single',None) else "?", "· build_masks no se llama con single")

print("\n=== 2. ¿el anillo aparece en TODAS las ventanas o solo en las que tocan borde? ===")
relleno = x[:,1].numpy()
toca = [i for i in range(10) if relleno[i].max() > 0]
with torch.no_grad(): mp = conv(x).numpy()
niv = np.median(mp[...,c,c], axis=(2,3), keepdims=True)
d = mp - niv
anillo = np.abs(d).copy(); anillo[...,c,c] = np.nan
for i in range(10):
    a = np.nanmean(anillo[i]); it = np.abs(d[i][...,c,c]).mean()
    print(f"   ventana #{i+1:>2} {'(TOCA borde)' if i in toca else '            '}"
          f"  anillo {a:.4f}   interior {it:.4f}   ratio {a/it:5.2f}x")

print("\n=== 3. ¿es el padding de CEROS? mismo modelo, relleno replicate ===")
with torch.no_grad():
    rep = F.conv2d(F.pad(x, (b,)*4, mode="replicate"), conv.weight, conv.bias).numpy()
for nom, t in (("ZEROS (el real)", mp), ("REPLICATE", rep)):
    nv = np.median(t[...,c,c], axis=(2,3), keepdims=True); dd = t - nv
    an = np.abs(dd).copy(); an[...,c,c] = np.nan
    print(f"   {nom:>16}: anillo {np.nanmean(an):.4f}   interior {np.abs(dd[...,c,c]).mean():.4f}"
          f"   ratio {np.nanmean(an)/np.abs(dd[...,c,c]).mean():5.2f}x")

print("\n=== 4. ¿y el canal de RELLENO cuanto aporta al anillo? ===")
for ch, nom in ((0,"vista"), (1,"relleno")):
    xc = torch.zeros_like(x); xc[:,ch] = x[:,ch]
    with torch.no_grad():
        t = F.conv2d(xc, conv.weight, None, conv.stride, conv.padding).numpy()
    an = np.abs(t).copy(); an[...,c,c] = np.nan
    print(f"   solo {nom:>8}: |anillo| {np.nanmean(an):.4f}")
