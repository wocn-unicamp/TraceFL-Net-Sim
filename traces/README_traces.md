# Pipeline de traces

Todo se ejecuta desde `TraceFL-Net-Sim/traces/`.

```
traces/
├── sys/                  traces originales de LEAF (entrada de todo)
├── sys_gen/              traces regenerados, para revisar antes de copiar a sys/
├── sys_bimodal/          traces con capacidad y tiempo de computo
├── net/                  salida del simulador Go (una por seed)
├── net_join/             traces de net/ unidos por tipo de simulacion
└── figures/
    ├── grouped_cdfs/     CDFs de carga (GFLOPs)
    ├── bimodal_cdfs/     CDFs de tiempo de computo
    └── net_grouped_cdfs/ CDFs de tiempo desde net_join/
```

---

## Flujo completo

```
                    sys/  (LEAF, 8 columnas sin cabecera)
                      |
      ┌───────────────┼────────────────┐
      |               |                |
gen_traces_*.py   plot_verificar   generate_bimodal_traces.py
      |            _traces.py              |
      v               v                    v
   sys_gen/     figures/grouped_cdfs   sys_bimodal/  (10 columnas)
      |                                    |
  (copiar a sys/)                    plot_bimodal_traces.py
                                           |
                                           v
                                  figures/bimodal_cdfs/

           simulador Go  ->  net/  ->  join_traces.py  ->  net_join/
                                                              |
                                                 plot_cdf_computation_time.py
                                                              |
                                                              v
                                                  figures/net_grouped_cdfs/
```

---

## 1. `gen_traces_shakespeare.py` y `gen_traces_femnist.py`

**Para que:** arreglan traces cuya CDF de carga no coincide con las demas al
variar el numero de clientes.

El problema en ambos casos era el mismo: la CDF de carga depende solo de
cuantas veces aparece cada cliente, y LEAF decide eso sorteando. Con pocos
sorteos el sorteo miente. La solucion es repartir las apariciones en vez de
sortearlas, fijandolas de antemano.

| | shakespeare | femnist |
|---|---|---|
| Sintoma | c=1..10 no coincidian con c=20 | solo c=50 se despegaba (4.46 pp) |
| Causa | 46 clientes y cargas de 0.68 a 40.8 GFLOPs | c=50 usaba 50 de los 184 clientes |
| Referencia | trace de c=20 | trace de c=30 (pool completo) |
| Pesos | proporcionales a la referencia | uniformes sobre los 184 |
| Resultado | 0.3 a 1.1 pp (2.4 con c=1) | 0.00 pp |

```bash
python gen_traces_shakespeare.py    # genera c = 2,3,4,5,8,10
python gen_traces_femnist.py        # genera c = 50
```

Escriben en `sys_gen/`. Revisa las CDFs y **copia a `sys/` manualmente** si
estan bien:

```bash
cp sys_gen/*.csv sys/
```

Para regenerar otros valores de c, edita la lista `CLIENTES` al principio del
archivo.

---

## 2. `plot_verificar_traces.py`

**Para que:** CDFs agrupadas de la carga computacional en GFLOPs. Es la
verificacion de que los traces de `sys/` son coherentes entre si.

```bash
python plot_verificar_traces.py
```

- Lee `sys/sys_metrics_*.csv`, columna 7 (`local_computations`), dividida por 1e9
- Escribe `figures/grouped_cdfs/cdf_<dataset>_<algoritmo>.png`
- FedAvg: una figura por dataset con una curva por c
- Minibatch: una figura por dataset y numero de clientes, una curva por mb

**Que esperar:** todas las curvas de una figura deben solaparse. Si una se
despega, ese trace usa un pool de clientes distinto y hay que regenerarlo con
los scripts del paso 1.

---

## 3. `generate_bimodal_traces.py`

**Para que:** asigna a cada cliente una capacidad de procesamiento y calcula
el tiempo de computo.

```bash
python generate_bimodal_traces.py
```

Todo se configura en el bloque de constantes al principio del archivo, no por
linea de comandos.

### El modelo

La bimodal esta en la **capacidad**, no en el tiempo:

```
C_i ~ N(0.5, 0.12^2)   el 50% de los clientes   (hardware lento)
C_i ~ N(1.5, 0.12^2)   el otro 50%              (hardware rapido)
       truncadas a [0.20, 1.80] GFLOP/s

T = local_computations / (C_i * 1e9)
```

Decisiones y por que:

- **Muestreo con rechazo, no `np.clip`.** Clip acumularia el 0.62% de las
  muestras exactamente en 0.20, y con 46 clientes eso es un cliente entero
  pegado al minimo generando el tiempo mas largo del trace.
- **Reparto 50/50 determinista** (barajar y cortar por la mitad), no Bernoulli
  independiente. Con 10 clientes, Bernoulli se sale del rango 40-60% el 65% de
  las veces.
- **Un mapa por dataset** (`--scope dataset`, por defecto). El mismo cliente
  tiene la misma capacidad en todos los traces, que es lo que mantiene
  comparables las CDFs de tiempo entre distintos c. Con `--scope file` cada
  archivo tiene su propio barajado y las curvas se separan.
- **Capacidad fija por cliente**, nunca regenerada por ronda.

Valores teoricos con truncamiento, utiles para validar: media 0.5021 y
desviacion 0.1173, no 0.5 y 0.12. Los modos estan separados 8.33 sigma, o sea
sin solapamiento.

### La flag `AMDAHL`

Es la constante `AMDAHL` al principio del archivo. Con `False` el speedup es 1
y el comportamiento es el de siempre.

Con `True`, 0.5 y 1.5 GFLOP/s se interpretan como la capacidad de **un
nucleo** y se multiplican por el speedup de Amdahl:

```
speedup = 1 / ((1 - p) + p / cores)
C_efectiva = C_base * speedup
```

| dataset | p | cores | speedup | modos efectivos | limites efectivos |
|---|---|---|---|---|---|
| femnist | 0.95 | 4 | 3.478 | 1.739 / 5.217 | [0.696, 6.261] |
| shakespeare | 0.40 | 4 | 1.429 | 0.714 / 2.143 | [0.286, 2.571] |

Multiplicar la capacidad escala tambien las desviaciones, que es lo que hacia
el `run_simulation.sh` original al multiplicar `std1` y `std2`. Los limites se
aplican a la capacidad base y se escalan con ella.

Se ajusta con las constantes `CORES`, `P_FEMNIST` y `P_SHAKESPEARE`.

El directorio de salida cambia solo: `sys_bimodal/` con `AMDAHL = False` y
`sys_bimodal_amdahl/` con `True`, para que una version no pise a la otra.

### Salida

`sys_bimodal/sys_metrics_*.csv`, **sin cabecera**, con las 8 columnas
originales intactas y dos anadidas al final:

| # | columna | ejemplo |
|---|---|---|
| 1 | client_id | `THE_LIFE_OF_TIMON_OF_ATHENS_LUCULLIUS` |
| 2 | round | `1` |
| 3 | hierarchy | *(vacia)* |
| 4 | num_samples | `338` |
| 5 | set | `train` |
| 6 | bytes_read | `3271488` |
| 7 | bytes_written | `3271488` |
| 8 | local_computations | `801298740` |
| 9 | **capacity_gflops** | `0.456588` |
| 10 | **time** | `1.754972848` |

Anadir al final y no poner cabecera es a proposito: cualquier lector
posicional que espere `usecols=[7]` sigue funcionando.

Ademas escribe:
- `sys_bimodal/capacities_<dataset>.csv` con el mapa cliente -> modo ->
  capacidad base -> speedup -> capacidad efectiva
- `sys_bimodal/plots/capacity_<dataset>.png` con el histograma y la gaussiana
  teorica superpuesta

### Constantes

```python
AMDAHL = False          # <-- la flag

CORES = 4
P_FEMNIST = 0.95
P_SHAKESPEARE = 0.4

MODE1_MEAN = 0.5        MODE1_STD = 0.12
MODE2_MEAN = 1.5        MODE2_STD = 0.12
MIN_CAPACITY = 0.20     MAX_CAPACITY = 1.80

SEED = 42
SCOPE = "dataset"       # o "file"
ENTRADA = sys/
SALIDA = sys_bimodal/ o sys_bimodal_amdahl/ segun AMDAHL
```

---

## 4. `plot_bimodal_traces.py`

**Para que:** CDFs agrupadas del tiempo de computo.

```bash
python plot_bimodal_traces.py
```

- Lee `sys_bimodal/sys_metrics_*.csv`, columna 9 (`time`)
- Escribe `figures/bimodal_cdfs/cdf_time_<dataset>_<algoritmo>.png`

**Que NO esperar:** la curva no sale bimodal, y eso es correcto. El tiempo
mezcla carga y capacidad, y la carga varia mas que la capacidad, asi que
aplasta los dos picos. La bimodalidad se verifica en
`sys_bimodal/plots/capacity_*.png`, no aqui.

Si generaste con `AMDAHL = True`, cambia `BIMODAL_DIR` a
`sys_bimodal_amdahl` al principio del script.

---

## 5. `join_traces.py`

**Para que:** el simulador Go produce un trace por seed. Este script los junta.

```bash
python join_traces.py
```

- Lee `net/metrics_network_*.csv` (con cabecera)
- Agrupa quitando `_seed_XXXX` del nombre
- Ordena poniendo primero todas las filas del round 1 de todas las seeds,
  luego las del round 2, etc.
- Escribe `net_join/` con el nombre sin la seed

Los CSV se leen con `dtype=str` para que los numeros se copien tal cual; si no,
pandas convertiria `0.000005291` en `5.291e-06`.

---

## 6. `plot_cdf_computation_time.py`

**Para que:** CDFs agrupadas del `computation-time` de los traces unidos.

```bash
python plot_cdf_computation_time.py
```

- Lee `net_join/metrics_network_*.csv`, columna `computation-time` por nombre
- Escribe `figures/net_grouped_cdfs/cdf_time_<dataset>_<algoritmo>.png`
- Ordena la leyenda numericamente (c=2, 3, 4, 5, 8, 10, 20...) en vez de
  alfabeticamente

Como los traces unidos mezclan las 5 seeds, cada curva agrega todas las
observaciones (cliente x ronda x seed) en una sola.

---

## Orden de ejecucion tipico

```bash
# 1. arreglar traces si alguna CDF de carga se despega
python gen_traces_shakespeare.py
python gen_traces_femnist.py
cp sys_gen/*.csv sys/

# 2. verificar
python plot_verificar_traces.py

# 3. capacidades y tiempo de computo (AMDAHL se elige dentro del archivo)
python generate_bimodal_traces.py

# 4. verificar
python plot_bimodal_traces.py

# 5. (simulador Go) -> net/

# 6. juntar seeds y graficar
python join_traces.py
python plot_cdf_computation_time.py
```

---

## Notas

**Numero de clientes.** El `c` del nombre son los clientes **por ronda**, no
los clientes distintos del trace. shakespeare c=20 tiene 46 clientes unicos;
femnist tiene 184 en casi todos sus traces.

**Reproducibilidad.** Todos los scripts usan semilla fija (42 por defecto).
Mismo trace + misma semilla + mismos parametros = mismo resultado. El orden de
los clientes se canonicaliza (ordenados alfabeticamente) antes de barajar,
porque el orden de un `set` de Python varia entre ejecuciones.

**Los originales no se tocan.** `gen_traces_*` escriben en `sys_gen/` y
`generate_bimodal_traces.py` en `sys_bimodal/`. La copia a `sys/` es manual y
deliberada.

**Formato del simulador Go.** El `data_processor.py` original entregaba al
simulador un CSV con cabecera y menos columnas. El pipeline nuevo mantiene el
formato de LEAF con dos columnas extra. Si el simulador Go necesita el formato
antiguo, hara falta un paso de conversion.
