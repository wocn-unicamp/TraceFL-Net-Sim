# 1 - Calcula ECDF y tu Smoothed CDF (bins=256, win=7).

# 2 -Ajusta mezcla lognormal desplazada con sklearn.mixture.GaussianMixture para 𝑘=1,2,3
# k=1,2,3 y un grid de loc.

# 3 Elige el mejor por SSE contra la Smoothed CDF (también imprime BIC).

# 4 Grafica ECDF + Smoothed CDF + CDF de la mezcla ganadora y muestra parámetros claros.


import numpy as np

loc = 714.9380166666666
w   = np.array([0.290092, 0.335257, 0.374651]) # pesos
mu  = np.array([7.186880, 8.083923, 6.879985]) # medias en log-espacio
sig = np.array([0.053885, 0.171617, 0.346232]) # desviaciones estándar en log-espacio

def sample_times(n, seed=123):
    rng = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=w)           # 1) componente
    z    = rng.normal(mu[comp], sig[comp])      # 2) normal en log-espacio
    y    = np.exp(z)                            # 3) lognormal
    x    = loc + y                              # 4) shift
    return x



# ejemplo
x = sample_times(1000)

print("Sample times (ms):", x)
print(x.mean(), np.median(x))