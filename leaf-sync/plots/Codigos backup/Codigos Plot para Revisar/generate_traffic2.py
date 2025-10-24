import numpy as np

# ===================== Parámetros del modelo (en FLOPs) =====================
loc = 1188270550.0
w   = np.array([0.040001, 0.641034, 0.318966])   # pesos
mu  = np.array([6.684612, 20.361297, 21.761275]) # medias en log-espacio (ln(FLOPs - loc))
sig = np.array([0.001000, 0.415943, 0.178851])   # desvíos estándar en log-espacio

upper_trunc = 5017145700.0  # cota superior (FLOPs)

# ===================== Capacidad del cliente =====================
FLOPS_PER_SEC = 5e9  # FLOPs/seg que puede procesar el cliente (p.ej., 5 GFLOPS)

def sample_flops(n, seed=123):
    rng  = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=w)            # 1) componente
    z    = rng.normal(mu[comp], sig[comp])       # 2) normal en log-espacio
    y    = np.exp(z)                              # 3) lognormal
    xF   = loc + y                                # 4) shift (FLOPs totales)
    xF   = np.minimum(xF, upper_trunc)            # 5) censura superior (1 línea)
    return xF

def flops_to_ms(flops, flops_per_sec=FLOPS_PER_SEC):
    return (flops / flops_per_sec) * 1e3         # FLOPs → ms

# ejemplo
fl = sample_flops(1000)
t_ms = flops_to_ms(fl, FLOPS_PER_SEC)

# Opt 2: Descarta valores > upper_trunc (no garantiza n)
# fl = fl[fl <= upper_trunc]
# t_ms = flops_to_ms(fl, FLOPS_PER_SEC)

print("Sample times (ms):", t_ms)
print(t_ms.mean(), np.median(t_ms))
