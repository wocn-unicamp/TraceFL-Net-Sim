# TraceFL-Net-Sim

**Trace-driven Federated-Learning Network Simulator**

[![Go](https://img.shields.io/badge/Go-%E2%89%A5%201.25-00ADD8?logo=go&logoColor=white)](go.mod)
[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](requirements.txt)
[![DOI](https://img.shields.io/badge/DOI-10.5753%2Fwperformance.2025.9221-blue)](https://doi.org/10.5753/wperformance.2025.9221)

TraceFL-Net-Sim is a trace-driven discrete-event simulator for evaluating how federated learning (FL) workloads interact with background traffic in bandwidth-constrained access networks. It combines three components:

- **`leaf-sync/`**: a modified version of the LEAF benchmark that runs the FL applications and records, for every client and round, the computational demand (FLOP/round) and the size of the model update;
- **`traces/`**: scripts that assign a processing capacity to each client and convert its computational demand into local computation time;
- **`trace_driven_simulator/`**: a discrete-event simulator, written in Go, of the access network in which the model updates compete with background traffic.

This repository accompanies the manuscript *Probabilistic Modeling and Performance Analysis of Federated Learning Traffic over Access Networks* (O. J. Ciceri-Coral, M. A. Guerra Pedroso, D. M. da Cunha, N. L. S. da Fonseca and C. A. Astudillo-Trujillo), submitted to the Journal of Internet Services and Applications (JISA), which extends our [WPerformance 2025 paper](https://doi.org/10.5753/wperformance.2025.9221).

> [!NOTE]
> The script that converts the computational demand reported by LEAF (FLOP/round) into local computation times, Stage 3 of the pipeline in the paper, is [`traces/convert_flops_to_trace.py`](traces/convert_flops_to_trace.py). It is described in [Stage 3](#stage-3-computation-time-of-each-client).

> [!TIP]
> The converted traces used in the paper are included in `traces/sys_bimodal_amdahl/`, so the network results can be reproduced without running LEAF. See [Quick start](#quick-start).

## Pipeline

| Stage (Fig. 2 of the paper) | What it does | Code | Output |
|---|---|---|---|
| 1–2 | Runs the FL applications and records the system metrics | `leaf-sync/paper_experiments/` | `traces/sys/` |
| 3 | Assigns client capacities and converts FLOP/round into computation time | `traces/convert_flops_to_trace.py` | `traces/sys_bimodal_amdahl/` |
| 4 | Simulates the FL and background traffic, five seeds per configuration | `run_simulation.sh`, `trace_driven_simulator/` | `traces/net/` |
| 5 | Merges the seeds and computes the network metrics and figures | `traces/join_traces.py`, `traces/plot_*.py` | `traces/net_join/`, `traces/figures/` |

The FL workflow is synchronous and round-based: in each round, the central server (CS) waits for the model updates of all participating clients before aggregating them, so the last update to arrive determines the duration of the round.

## Quick start

Requirements: Linux with `bash` and `bc`, [Go](https://go.dev/dl/) ≥ 1.25, and Python 3 with the packages in `requirements.txt`.

```bash
git clone https://github.com/wocn-unicamp/TraceFL-Net-Sim.git
cd TraceFL-Net-Sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stage 3 (optional): the converted traces used in the paper are already in traces/sys_bimodal_amdahl/
(cd traces && python convert_flops_to_trace.py)

# Stage 4: network simulation, five seeds per configuration -> traces/net/
./run_simulation.sh

# Stage 5: merge the seeds and plot -> traces/net_join/, traces/figures/
cd traces
python join_traces.py
python plot_cdf_computation_time.py
python plot_fl_training_time.py
```

`run_simulation.sh` builds the simulator if Go is installed. On a server without Go, build the binary on another machine with `GOOS=linux GOARCH=amd64 go build -o sim_bin trace_driven_simulator/main.go` and run `SIM_BIN=/path/to/sim_bin ./run_simulation.sh`.

## Repository structure

```text
TraceFL-Net-Sim/
├── run_simulation.sh              # Stage 4: runs all network simulations
├── params/                        # Simulation parameter files
├── requirements.txt               # Python packages for stages 3–5
├── go.mod, go.sum                 # Go module (Go ≥ 1.25)
├── leaf-sync/                     # Stages 1–2: modified LEAF (see leaf-sync/README.md)
│   ├── paper_experiments/         #   femnist.sh and shakespeare.sh are used in the paper
│   ├── plots_2026/                #   accuracy and computational-demand analyses
│   ├── requirements.txt           #   LEAF environment
│   └── LICENSE.md                 #   original LEAF license
├── traces/                        # Stages 3 and 5 (see traces/README.md)
│   ├── convert_flops_to_trace.py  #   capacity assignment, FLOP/round → computation time
│   ├── join_traces.py             #   merges the outputs of the five seeds
│   ├── plot_*.py                  #   validation and result figures
│   ├── sys/                       #   original LEAF system traces
│   ├── sys_bimodal/               #   heterogeneous capacities, without Amdahl scaling
│   ├── sys_bimodal_amdahl/        #   heterogeneous capacities with Amdahl scaling (paper)
│   ├── net/                       #   simulator outputs, one file per seed
│   ├── net_join/                  #   simulator outputs merged across seeds
│   ├── stat/, normalize/          #   auxiliary data of the analysis scripts
│   └── figures/                   #   generated figures
├── trace_driven_simulator/        # Stage 4: discrete-event simulator
│   ├── main.go                    #   entry point
│   ├── data_processor.py          #   prepares the simulator input (called by run_simulation.sh)
│   ├── internal/simulator/        #   models, event handling and queues
│   └── packages/writer/           #   output writers
├── speed_up_eval/                 # Monte Carlo analysis (Section 5.3)
└── figures/
```

## Stages 1 and 2. FL workload with LEAF

The paper uses FEMNIST with a CNN (`leaf-sync/paper_experiments/femnist.sh`) and Shakespeare with an LSTM (`shakespeare.sh`); the other scripts in that folder are not used. This step is only needed to generate new workloads. LEAF depends on TensorFlow 1.x, which requires Python 3.6, so it needs its own environment:

```bash
conda create -n leaf-sync python=3.6 -y
conda activate leaf-sync
pip install -r leaf-sync/requirements.txt
cd leaf-sync/paper_experiments
bash femnist.sh        # or: bash shakespeare.sh
```

Copy the resulting system-metrics files (`sys_metrics_*.csv`) to `traces/sys/`, the input of Stage 3.

## Stage 3. Computation time of each client

### Processing capacity

Each client receives a base processing capacity that represents a low- or a high-capacity device:

```math
C_i^{\mathrm{base}} \sim
\begin{cases}
\mathcal{N}\left(0.5,\ 0.12^{2}\right) & \text{for low-capacity clients,}\\
\mathcal{N}\left(1.5,\ 0.12^{2}\right) & \text{for high-capacity clients,}
\end{cases}
\qquad 0.20 \le C_i^{\mathrm{base}} \le 1.80 \ \text{GFLOP/s}.
```

Half of the clients belong to each group. Values outside [0.20, 1.80] GFLOP/s are redrawn (rejection sampling), so each group follows a truncated normal distribution without artificial probability mass at the limits. Each client keeps its capacity in all rounds.

### Amdahl scaling

The base capacity is scaled by the Amdahl speedup $`S_d`$ of application $`d`$, which depends on its parallelizable fraction $`P_d`$ and on the number of cores $`N_c`$ (Eqs. 7 and 8 of the paper):

```math
S_d = \frac{1}{(1-P_d) + \dfrac{P_d}{N_c}},
\qquad
C_{i,d}^{\mathrm{eff}} = S_d \, C_i^{\mathrm{base}}.
```

| Application | $`P_d`$ | $`N_c`$ | $`S_d`$ | Range of $`C_{i,d}^{\mathrm{eff}}`$ (GFLOP/s) | Mean $`C_{i,d}^{\mathrm{eff}}`$, low / high group (GFLOP/s) |
|---|---:|---:|---:|---:|---:|
| FEMNIST (CNN) | 0.95 | 4 | 3.478 | 0.70–6.26 | 1.75 / 5.21 |
| Shakespeare (LSTM) | 0.40 | 4 | 1.429 | 0.29–2.57 | 0.72 / 2.14 |

The parallelizable fractions are modeling assumptions: the convolutions of the CNN parallelize well across cores, whereas the recurrent dependencies of the LSTM limit parallel execution.

### Local computation time

The computation time of client $`i`$ in round $`r`$ follows from its computational demand $`X_{i,r}`$, in FLOP/round as reported by LEAF, and its effective capacity in GFLOP/s (Eq. 9 of the paper):

```math
T_{i,r} = \frac{X_{i,r}}{C_{i,d}^{\mathrm{eff}} \times 10^{9}} \quad [\mathrm{s}],
```

where the factor $`10^{9}`$ converts GFLOP/s into FLOP/s. For example, a FEMNIST round of 2 GFLOP takes 1.15 s on a client with $`C_i^{\mathrm{base}} = 0.5`$ ($`C_{i,d}^{\mathrm{eff}} = 1.74`$ GFLOP/s) and 0.38 s on a client with $`C_i^{\mathrm{base}} = 1.5`$ ($`C_{i,d}^{\mathrm{eff}} = 5.22`$ GFLOP/s). $`T_{i,r}`$ is the instant at which the model update of the client enters its transmission queue in the simulator. Since the bimodal model applies to the capacity and not to the time, the distribution of $`T_{i,r}`$ is not necessarily bimodal.

### Trace format

`convert_flops_to_trace.py` reads `traces/sys/` without modifying it and writes the converted traces to `traces/sys_bimodal_amdahl/` (or `traces/sys_bimodal/` without Amdahl scaling), together with the client-to-capacity mapping and plots of the capacity distributions (Figure 8 of the paper). The files have no header; the converted traces keep the eight LEAF fields and append two:

| # | Field | Content |
|---:|---|---|
| 1 | `client_id` | Client identifier |
| 2 | `round` | Training round $`r`$ |
| 3 | `hierarchy` | Client group (LEAF field) |
| 4 | `num_samples` | Number of local samples |
| 5 | `set` | Data partition |
| 6 | `bytes_read` | Bytes received from the server |
| 7 | `bytes_written` | Bytes sent to the server (model-update size) |
| 8 | `local_computations` | Computational demand $`X_{i,r}`$ (FLOP/round) |
| 9 | `capacity_gflops` | Capacity assigned to the client (GFLOP/s); $`C_{i,d}^{\mathrm{eff}}`$ when Amdahl scaling is enabled (added) |
| 10 | `time` | Computation time $`T_{i,r}`$ (s) (added) |

## Stage 4. Network simulation

TraceFL-Net-Sim reproduces the transmission of Ethernet frames from the FL clients and the background sources to the CS:

- **Topology.** Each FL client is connected to the access device by its own link, every traffic source has an independent input queue, and all frames leave through a shared output link to the CS. Queues are served FCFS and have infinite buffers, and the path from the switch to the CS has a constant delay.
- **FL traffic.** A model update enters the queue of its client at the end of local computation ($`T_{i,r}`$). It is fragmented into Ethernet frames (1500-byte MTU plus an 18-byte header) and reassembled when all its frames reach the CS.
- **Events.** Frame arrival at an input queue, forwarding to the output queue, enqueuing at the output queue, and departure towards the CS.
- **Background traffic.** Three profiles, each parameterized to keep the configured mean offered load. In the multi-traffic configuration used in the paper, the three run concurrently with equal shares of the load:

| Profile | Represents | Inter-arrival times | Frame size |
|---|---|---|---|
| Poisson | Web-like traffic | Exponential | 64–1518 B |
| Pareto | Bursty multimedia-like traffic | Bounded Pareto (heavy-tailed) | 64–1518 B |
| CBR | VoIP-like traffic | Constant (periodic) | 70 B |

### Frame losses

Each transmission attempt fails independently with probability $`p`$, the frame-loss rate (the simulator option `-transmission-success-rate` equals $`1-p`$); in the paper, losses are applied at the server-side queue. A lost frame is not discarded: after a backoff it is re-inserted into the queue for a new attempt, which is repeated until the frame reaches the CS. The backoff of the $`n`$-th retransmission is

```math
B_n = 2^{n}\,\xi, \qquad \xi \sim \mathcal{U}(0.016,\ 0.064)\ \mathrm{s}.
```

A frame therefore needs $`1/(1-p)`$ attempts on average (1.25 for $`p = 0.20`$). Because the buffers are infinite, losses come only from this model and never from buffer overflow.

### Delay, round duration and training time

The delay of the model update of client $`i`$ in round $`r`$ runs from the departure of its first frame from the client, $`t_{i,r}^{\mathrm{dep}}`$, until the arrival of its last frame at the CS, $`t_{i,r}^{\mathrm{CS}}`$:

```math
D_{i,r} = t_{i,r}^{\mathrm{CS}} - t_{i,r}^{\mathrm{dep}}.
```

With synchronous aggregation, round $`r`$ starts at $`s_r`$ and ends when the last update of the set $`\mathcal{N}_F`$ of participating clients reaches the CS. Rounds run back to back, so the total training time over $`N_R`$ rounds is the sum of the round durations:

```math
R_r = \max_{i \in \mathcal{N}_F} t_{i,r}^{\mathrm{CS}} - s_r,
\qquad
T^{\mathrm{total}} = \sum_{r=1}^{N_R} R_r .
```

Figures 14 and 15 split each round into the computation time of the client whose update arrives last, $`T_{i_r^{*},r}`$ with $`i_r^{*} = \arg\max_{i \in \mathcal{N}_F} t_{i,r}^{\mathrm{CS}}`$, and the remainder $`R_r - T_{i_r^{*},r}`$, attributed to communication.

### Parameters

The values of the discrete-event evaluation (Table 4 of the paper) are set in `run_simulation.sh` and `params/`; the capacity parameters are set in `traces/convert_flops_to_trace.py` (see [Stage 3](#stage-3-computation-time-of-each-client)).

| Parameter | Value |
|---|---|
| FL-device link capacity | 1.5 Gb/s |
| Shared output-link capacity | 2.25 Gb/s |
| Aggregate background load | 67 % of the output-link capacity (≈ 1.5 Gb/s), ≈ 0.50 Gb/s per profile |
| Participating clients, $`\beta = 1.0`$ | FEMNIST {3, 5, 10, 20, 30, 50}; Shakespeare {2, 3, 4, 5, 8, 10, 20} |
| Mini-batch fraction $`\beta`$ | {0.2, 0.4, 0.5, 0.6, 0.8, 0.9}, with 20 FEMNIST or 10 Shakespeare clients |
| Frame-loss rate $`p`$ | 0 (Figs. 11, 12, 14); 0, 0.05, 0.10, 0.15, 0.20 with 20 FEMNIST or 10 Shakespeare clients and $`\beta = 1.0`$ (Figs. 13, 15) |
| Model-update size | FEMNIST 26.4 MB; Shakespeare 32.72 MB |
| Training rounds $`N_R`$ | FEMNIST 500; Shakespeare 50 |
| Random seeds | 42, 1337, 2026, 8888, 9999 |

`run_simulation.sh` calls `trace_driven_simulator/data_processor.py` to prepare the simulator input from the converted traces and runs up to five simulations in parallel. Runs whose output already exists are skipped, so an interrupted campaign can be resumed.

## Stage 5. Aggregation and analysis

The seeds change only the background traffic and the frame losses; the FL traces and the client capacities are the same in all runs. `join_traces.py` merges the five seeds of each experiment, removing the seed suffix from the file name and keeping all observations. The CDFs of Figures 11–13 pool the five seeds, and the training times of Figures 14–15 are means with 95 % confidence intervals across seeds.

All scripts are in `traces/`, and paths are relative to it:

| Script | Input → output | Purpose |
|---|---|---|
| `plot_verificar_traces.py` | `sys/` → `figures/grouped_cdfs/` | CDFs of the computational demand, to check the LEAF traces before conversion |
| `plot_bimodal_traces.py` | `sys_bimodal/` or `sys_bimodal_amdahl/` → `figures/bimodal_cdfs/` or `figures/bimodal_cdfs_amdahl/` | CDFs of the computation times after the capacity assignment |
| `join_traces.py` | `net/metrics_network_*.csv` → `net_join/` | Merges the seeds of each experiment |
| `plot_cdf_computation_time.py` | `net_join/` → `figures/net_grouped_cdfs/` | CDFs of the computation and model-update arrival times |
| `plot_fl_training_time.py` | network traces → `figures/fl_training_time/` | Total training time and its computation and communication components |

## Monte Carlo analysis

`speed_up_eval/` contains the complementary analysis of Section 5.3, which samples the probabilistic models of Section 4 instead of running the network simulator. In each replication, the computational demand and the capacity of every client are sampled, the round lasts as long as the slowest client (no communication delay), and the average offered load is the volume of model updates per round divided by the mean round duration:

```math
R^{\mathrm{MC}} = \max_{i \in \mathcal{N}_F} T_i,
\qquad
\bar{L} = \frac{8\,\lvert\mathcal{N}_F\rvert\,S}{\mathbb{E}\left[R^{\mathrm{MC}}\right]} \ [\mathrm{bit/s}],
\qquad
T^{\mathrm{total}} = N_R^{\mathrm{target}}\,\mathbb{E}\left[R^{\mathrm{MC}}\right],
```

where $`S`$ is the model-update size in bytes and $`N_R^{\mathrm{target}}`$ is the number of rounds needed to reach the target accuracy, taken from the accuracy curves (Figures 9–10). For example, 20 FEMNIST clients ($`S`$ = 26.4 MB) with a mean round of 6.6 s give $`\bar{L}`$ = 8 × 20 × 26.4 × 10⁶ / 6.6 s = 640 Mb/s and, with 400 rounds, $`T^{\mathrm{total}}`$ = 44 min (Table 5). The capacity profiles (homogeneous, bimodal, Gaussian and log-normal) share a mean of 1 GFLOP/s, are truncated to [0.25, 1.75] GFLOP/s and are applied before Amdahl scaling; each analysis uses 10⁶ replications.

## Reproducing the figures and tables

<!-- Verify each row against the scripts that produced the figures of the manuscript. -->

| Paper | Content | Code |
|---|---|---|
| Figs. 5–7, Tables 2–3 | Fitted distributions of the computational demand | `leaf-sync/plots_2026/` |
| Fig. 8 | Effective capacities of the clients | `traces/convert_flops_to_trace.py` |
| Figs. 9–10 | Accuracy vs. training rounds | `leaf-sync/plots_2026/` |
| Figs. 11–13 | CDFs of computation and model-update arrival times | `traces/plot_cdf_computation_time.py` |
| Figs. 14–15 | Total training time | `traces/plot_fl_training_time.py` |
| Figs. 16–19, Tables 5–6 | Monte Carlo analysis | `speed_up_eval/` |

## Scope and limitations

- **Orchestration.** Aggregation waits for all participating clients, so slow clients are never excluded. Deadline-based aggregation, in which late clients become stragglers, is future work.
- **Buffers.** Queues are infinite, so no frame is lost to buffer overflow; losses come only from the probabilistic model. Finite buffers are future work.
- **Beyond the access network.** The switch–CS path has a constant delay, so the results isolate contention in the access network. WAN delay variability (RTT, jitter, further bottlenecks) is not modeled and would lengthen the rounds.
- **Access technology.** Link capacities and FCFS queues abstract a generic shared access network. Technology-specific MAC scheduling and bandwidth allocation (e.g., PON, 5G, Wi-Fi) are not modeled, so the delays are not predictions for a specific technology.
- **Validation.** Traffic generation was validated against the discrete-event optical-network simulator of Ciceri et al. (IEEE Network, 2022). An independent validation of the latency measurements against ns-3 is future work.

## Citation

If you use TraceFL-Net-Sim, please cite:

```bibtex
@unpublished{ciceri2025probabilistic,
  author = {Ciceri-Coral, Oscar Jaime and Guerra Pedroso, Marco Aurelio and da Cunha, Diogo Maciel and da Fonseca, Nelson Luis Saldanha and Astudillo-Trujillo, Carlos Alberto},
  title  = {Probabilistic Modeling and Performance Analysis of Federated Learning Traffic over Access Networks},
  note   = {Submitted to the Journal of Internet Services and Applications (JISA)}
}

@inproceedings{cunha2025avaliaccao,
  title     = {Avalia{\c{c}}{\~a}o de Desempenho de Aplica{\c{c}}{\~o}es de Aprendizado Federado em Redes de Acesso Compartilhadas},
  author    = {Cunha, Diogo M. and Guerra, Marco A. and Ciceri, Oscar J. and da Fonseca, Nelson L. S. and Astudillo, Carlos A.},
  booktitle = {Anais do XXIV Workshop em Desempenho de Sistemas Computacionais e de Comunica{\c{c}}{\~a}o (WPerformance)},
  pages     = {121--132},
  year      = {2025},
  publisher = {SBC},
  doi       = {10.5753/wperformance.2025.9221}
}
```

## License

`leaf-sync/` is derived from [LEAF](https://github.com/TalwalkarLab/leaf) (S. Caldas et al., *LEAF: A Benchmark for Federated Settings*, arXiv:1812.01097, 2018) and keeps its original license ([`leaf-sync/LICENSE.md`](leaf-sync/LICENSE.md)). The rest of the repository is distributed under the terms of the [`LICENSE`](LICENSE) file.
