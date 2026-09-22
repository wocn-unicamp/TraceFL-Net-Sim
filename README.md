# TraceFL-Net-Sim

**TraceFL-Net-Sim** is a trace-driven discrete-event simulator for evaluating how Federated Learning (FL) workloads interact with background traffic in bandwidth-constrained access networks.

The project combines three main components:

1. a modified synchronous FL workflow based on LEAF;
2. a trace-processing pipeline that converts per-client computational demand into local computation times;
3. a Go-based discrete-event network simulator.

This repository accompanies the manuscript:

> **Probabilistic Modeling and Performance Analysis of Federated Learning Traffic over Access Networks**

---

## Overview

The complete experimental workflow is:

```text
FL experiments
  leaf-sync/
      |
      | per-client system metrics
      | FLOP/round, model-update size, client, round
      v
Trace processing
  traces/
      |
      | computational-demand validation
      | client-capacity assignment
      | FLOP -> computation-time conversion
      v
Processed FL traces
      |
      v
Network simulation
  run_simulation.sh
  trace_driven_simulator/
      |
      | per-seed network metrics
      v
Trace aggregation and analysis
  traces/net/
  traces/net_join/
  traces/figures/
```

The FL and network stages are intentionally separated. The FL execution produces the computational workload, while TraceFL-Net-Sim evaluates how the corresponding model updates are affected by the communication network.

The evaluated FL workflow is synchronous and round-based: the central server waits for the model updates from all participating clients before completing the current training round.

---

## Repository Structure

```text
TraceFL-Net-Sim/
├── README.md
├── requirements.txt
├── go.mod
├── go.sum
├── run_simulation.sh
│
├── figures/
├── params/
├── speed_up_eval/
│
├── leaf-sync/
│   ├── baseline/
│   ├── benchmark_comp_capacity/
│   ├── data/
│   ├── datasets/
│   ├── docs/
│   ├── generador_trafico/
│   ├── models/
│   ├── paper_experiments/
│   ├── plots/
│   ├── plots_2026/
│   ├── results/
│   ├── LICENSE.md
│   ├── README.md
│   └── requirements.txt
│
├── traces/
│   ├── convert_flops_to_trace.py
│   ├── join_traces.py
│   ├── plot_bimodal_traces.py
│   ├── plot_cdf_computation_time.py
│   ├── plot_fl_training_time.py
│   ├── plot_verificar_traces.py
│   ├── sys/
│   ├── sys_bimodal/
│   ├── sys_bimodal_amdahl/
│   ├── net/
│   ├── net_join/
│   ├── stat/
│   ├── normalize/
│   └── figures/
│       ├── bimodal_cdfs/
│       ├── bimodal_cdfs_amdahl/
│       ├── fl_training_time/
│       ├── grouped_cdfs/
│       ├── net_grouped_cdfs/
│       └── net_grouped_cdfs_diogo/
│
└── trace_driven_simulator/
    ├── data_processor.py
    ├── main.go
    ├── internal/
    │   └── simulator/
    │       ├── functions.go
    │       ├── models.go
    │       └── queues/
    └── packages/
        └── writer/
            ├── constants.go
            ├── functions.go
            └── models.go
```

The main directories are described below.

---

## 1. FL Workload Generation

The first stage is performed under:

```text
leaf-sync/
```

This directory contains the modified LEAF-based environment used to execute the FL applications and collect the workload information required by TraceFL-Net-Sim.

The FL execution produces system metrics for each client and training round, including:

- client identifier;
- training round;
- number of local samples;
- bytes read;
- bytes written;
- local computational demand (`FLOP/round`).

The paper experiment scripts are located in:

```text
leaf-sync/paper_experiments/
```

Current experiment scripts include:

```text
femnist.sh
shakespeare.sh
sent140.sh
test.sh
```

The experiments considered in the associated manuscript primarily use:

- **FEMNIST with a CNN**;
- **Shakespeare with an LSTM**.

Additional analysis scripts are available under:

```text
leaf-sync/plots_2026/
```

These scripts include utilities for analyzing model accuracy, computational demand, computation time, target accuracy, and total computation time.

### Running an FL experiment

From the repository root:

```bash
cd leaf-sync/paper_experiments
```

For example:

```bash
bash femnist.sh
```

or:

```bash
bash shakespeare.sh
```

The system-metrics traces required by the next stage are placed under:

```text
traces/sys/
```

---

## 2. Original FL Traces

The directory

```text
traces/sys/
```

contains the original FL system traces used as input to the trace-processing stage.

The files preserve the LEAF system-metrics format and contain eight fields without a header:

| Position | Field |
|---:|---|
| 1 | `client_id` |
| 2 | `round` |
| 3 | `hierarchy` |
| 4 | `num_samples` |
| 5 | `set` |
| 6 | `bytes_read` |
| 7 | `bytes_written` |
| 8 | `local_computations` |

The `local_computations` field contains the per-client computational demand in FLOPs.

Before converting these traces into computation times, their computational-demand distributions can be inspected with:

```bash
cd traces
python plot_verificar_traces.py
```

This script reads the `local_computations` field and generates grouped cumulative distribution functions (CDFs) under:

```text
traces/figures/grouped_cdfs/
```

These figures are used as consistency checks when comparing workloads obtained from different FL configurations.

---

## 3. FLOP-to-Trace Conversion

The main conversion script is:

```text
traces/convert_flops_to_trace.py
```

This script converts the per-client computational demand reported by LEAF into local computation times. These times determine when each client's model update becomes available for transmission by the network simulator.

For client $i$ in training round $r$, the local computation time is calculated as:

$$
T_{i,r} =
\frac{X_{i,r}}
{C^{\mathrm{eff}}_{i,d} \times 10^9},
$$

where:

- $X_{i,r}$ is the computational demand reported by LEAF in `FLOP/round`;
- $C^{\mathrm{eff}}_{i,d}$ is the effective processing capacity of client $i$ for application $d$, expressed in GFLOP/s.

The script preserves the eight original LEAF fields and appends two additional columns:

| Position | Field | Description |
|---:|---|---|
| 9 | `capacity_gflops` | Effective processing capacity assigned to the client |
| 10 | `time` | Local computation time in seconds |

The original traces stored in `traces/sys/` are not modified.

### Heterogeneous Processing Capacities

Client processing capacities are modeled using a bimodal distribution:

$$
C_i^{\mathrm{base}} \sim
\begin{cases}
\mathcal{N}(0.5,\,0.12^2), & \text{low-capacity clients}, \\
\mathcal{N}(1.5,\,0.12^2), & \text{high-capacity clients}.
\end{cases}
$$

Half of the clients are assigned to each group. Samples are restricted to:

$$
0.20 \leq C_i^{\mathrm{base}} \leq 1.80
\quad \text{GFLOP/s}.
$$

A processing capacity is assigned once to each client and remains fixed across training rounds.

The script uses rejection sampling rather than clipping values at the distribution limits, avoiding artificial probability mass at the boundaries.

### Amdahl Scaling

For the journal experiments, the base processing capacity is scaled according to Amdahl's law to represent the different degrees of parallelism of the evaluated applications.

For application $d$, the speedup is:

$$
S_d =
\frac{1}
{(1-P_d)+\frac{P_d}{N_c}},
$$

where:

- $P_d$ is the parallelizable fraction of the application;
- $N_c$ is the number of processing cores.

The effective processing capacity is then:

$$
C^{\mathrm{eff}}_{i,d}
=
C_i^{\mathrm{base}} S_d.
$$

The current experiment configuration uses:

| Application | Parallelizable fraction ($P_d$) | Cores ($N_c$) | Speedup ($S_d$) |
|---|---:|---:|---:|
| FEMNIST/CNN | 0.95 | 4 | 3.478 |
| Shakespeare/LSTM | 0.40 | 4 | 1.429 |

To generate the processed traces:

```bash
cd traces
python convert_flops_to_trace.py
```

With Amdahl scaling enabled, the generated traces are stored in:

```text
traces/sys_bimodal_amdahl/
```

The script also generates client-to-capacity mappings and plots of the resulting processing-capacity distributions.

---

## 4. Validation of Converted Traces

The `traces/` directory contains additional scripts used to inspect the generated workloads before running the network simulator.

### Computational-Demand Validation

```text
plot_verificar_traces.py
```

This script analyzes the original traces in `traces/sys/` and produces grouped CDFs of the computational demand.

Run:

```bash
python plot_verificar_traces.py
```

Output:

```text
traces/figures/grouped_cdfs/
```

### Computation-Time Validation

```text
plot_bimodal_traces.py
```

This script analyzes the local computation times obtained after assigning heterogeneous processing capacities.

Run:

```bash
python plot_bimodal_traces.py
```

Depending on the selected input directory, the generated figures are stored under:

```text
traces/figures/bimodal_cdfs/
```

or:

```text
traces/figures/bimodal_cdfs_amdahl/
```

The bimodal model applies to **processing capacity** rather than directly to computation time. Since computation time depends jointly on computational demand and processing capacity, the resulting computation-time distribution is not necessarily bimodal.

---

## 5. Network Simulator

The network simulator is implemented in Go under:

```text
trace_driven_simulator/
```

Its main structure is:

```text
trace_driven_simulator/
├── main.go
├── data_processor.py
├── internal/simulator/
│   ├── functions.go
│   ├── models.go
│   └── queues/
└── packages/writer/
    ├── constants.go
    ├── functions.go
    └── models.go
```

The components have the following roles:

- `main.go`: main simulator entry point;
- `internal/simulator/`: simulation models, event-processing functions, and queue implementation;
- `packages/writer/`: output structures and functions used to write simulation metrics;
- `data_processor.py`: auxiliary data-processing utility associated with the simulator workflow.

TraceFL-Net-Sim reproduces Ethernet-frame transmission from FL clients and background-traffic sources toward a central server.

The modeled network includes:

- FL-device access links;
- independent traffic-source queues;
- a shared output link;
- FCFS queue service;
- configurable link capacities;
- propagation delay;
- probabilistic frame losses;
- retransmissions;
- frame delivery and reassembly at the central server.

The simulator provides a technology-agnostic abstraction of a shared access network. Its purpose is to evaluate contention, queueing, and bottleneck effects rather than reproduce the MAC-layer scheduler of a specific access technology.

---

## 6. Background Traffic

TraceFL-Net-Sim supports three concurrent background-traffic profiles:

| Traffic profile | Representative service | Temporal behavior |
|---|---|---|
| Poisson | Web-like traffic | Exponentially distributed inter-arrival times |
| Pareto | Multimedia-like traffic | Bursty and heavy-tailed behavior |
| CBR | VoIP-like traffic | Periodic frame generation |

The background sources coexist with FL traffic and compete for the capacity of the shared network resources.

---

## 7. Running the Network Simulation

The top-level script:

```text
run_simulation.sh
```

is used to launch the network-simulation workflow.

From the repository root:

```bash
./run_simulation.sh
```

Simulation parameter files are stored under:

```text
params/
```

The network simulator uses the processed FL traces as workload input and produces one output file for each configured random seed.

The resulting network traces are stored under:

```text
traces/net/
```

These traces contain the network-level measurements used by the subsequent analysis scripts.

---

## 8. Combining Simulation Runs

The Go simulator produces a separate network trace for each seed.

The script:

```text
traces/join_traces.py
```

combines traces corresponding to the same experiment.

Run:

```bash
cd traces
python join_traces.py
```

Input:

```text
traces/net/metrics_network_*.csv
```

Output:

```text
traces/net_join/
```

The seed suffix is removed from the output experiment name, while observations from all seeds are preserved in the combined trace.

---

## 9. Network and Training-Time Analysis

Several scripts under `traces/` operate on the network-simulation outputs.

### Computation-Time CDFs

```text
plot_cdf_computation_time.py
```

This script reads the combined network traces and generates grouped CDFs of the `computation-time` field.

Run:

```bash
python plot_cdf_computation_time.py
```

Output:

```text
traces/figures/net_grouped_cdfs/
```

### FL Training Time

```text
plot_fl_training_time.py
```

This script processes the network traces to analyze FL training time after incorporating the simulated communication delays.

Run:

```bash
python plot_fl_training_time.py
```

Output:

```text
traces/figures/fl_training_time/
```

The `traces/figures/` directory currently contains:

```text
bimodal_cdfs/
bimodal_cdfs_amdahl/
fl_training_time/
grouped_cdfs/
net_grouped_cdfs/
net_grouped_cdfs_diogo/
```

---

## 10. Trace Directory Summary

The main data directories under `traces/` are:

| Directory | Purpose |
|---|---|
| `sys/` | Original FL system traces |
| `sys_bimodal/` | Traces with heterogeneous capacities without Amdahl scaling |
| `sys_bimodal_amdahl/` | Traces with heterogeneous capacities and Amdahl scaling |
| `net/` | Network-simulator outputs separated by seed |
| `net_join/` | Network outputs combined across seeds |
| `stat/` | Auxiliary statistical data used by the analysis workflow |
| `normalize/` | Auxiliary normalized data used by processing and analysis scripts |
| `figures/` | Validation and result figures |

The `traces/` directory also contains its own README with additional implementation details.

---

## 11. Complementary Computational Analysis

The directory:

```text
speed_up_eval/
```

contains complementary computational analyses used to study processing-capacity effects independently from the detailed discrete-event network simulation.

These analyses support the evaluation of quantities such as:

- client computation time;
- training-round duration;
- computational heterogeneity;
- offered FL network load.

---

## Installation

The repository contains both Python and Go components.

### Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the root Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The modified LEAF component also contains its own dependency specification:

```text
leaf-sync/requirements.txt
```

Depending on the LEAF version and local environment, a separate Python environment may be preferable for `leaf-sync/`.

### Go Environment

From the repository root:

```bash
go mod download
```

Check the installed versions:

```bash
python --version
go version
```

---

## Typical End-to-End Execution

A typical experiment follows the sequence below.

### Step 1 — Run the FL Application

```bash
cd leaf-sync/paper_experiments
bash femnist.sh
```

or:

```bash
bash shakespeare.sh
```

Place the required system-metrics traces under:

```text
traces/sys/
```

### Step 2 — Inspect the Original Computational-Demand Traces

From the repository root:

```bash
cd traces
python plot_verificar_traces.py
```

### Step 3 — Convert FLOP/round into Local Computation Time

```bash
python convert_flops_to_trace.py
```

For the current Amdahl-enabled configuration, the processed traces are written to:

```text
sys_bimodal_amdahl/
```

### Step 4 — Inspect the Generated Computation-Time Traces

```bash
python plot_bimodal_traces.py
```

Configure the script to use `sys_bimodal_amdahl/` when analyzing the Amdahl-enabled traces.

### Step 5 — Run TraceFL-Net-Sim

Return to the repository root:

```bash
cd ..
./run_simulation.sh
```

Per-seed network outputs are written under:

```text
traces/net/
```

### Step 6 — Combine the Network Outputs

```bash
cd traces
python join_traces.py
```

Combined traces are written to:

```text
traces/net_join/
```

### Step 7 — Generate the Final Analyses

```bash
python plot_cdf_computation_time.py
python plot_fl_training_time.py
```

The resulting figures are stored under:

```text
traces/figures/
```

---

## Modeling Scope

TraceFL-Net-Sim is intended to evaluate the interaction between FL workload characteristics and communication-network constraints.

The current model assumes:

- synchronous, round-based FL;
- fixed client processing capacity across training rounds;
- computation time derived from per-round computational demand;
- heterogeneous client processing capacities;
- FCFS queueing;
- configurable background traffic;
- configurable probabilistic frame loss and retransmission;
- a shared access-network bottleneck.

Technology-specific MAC scheduling and bandwidth-allocation mechanisms are outside the current abstraction. Therefore, the reported delays characterize the configured shared access-network scenario rather than a specific 5G, Wi-Fi, or PON implementation.

---

## Citation

If you use TraceFL-Net-Sim in your research, please cite the corresponding publication.

Previous conference version:

```bibtex
@inproceedings{cunha2025avaliaccao,
  title={Avalia{\c{c}}{\~a}o de Desempenho de Aplica{\c{c}}{\~o}es de Aprendizado Federado em Redes de Acesso Compartilhadas},
  author={Cunha, Diogo M. and Guerra, Marco A. and Ciceri, Oscar J. and da Fonseca, Nelson L. S. and Astudillo, Carlos A.},
  booktitle={Workshop em Desempenho de Sistemas Computacionais e de Comunica{\c{c}}{\~a}o (WPerformance)},
  pages={121--132},
  year={2025},
  organization={SBC}
}
```

The citation for the extended journal article will be added after publication.

---

## License

The modified LEAF component includes its corresponding license under:

```text
leaf-sync/LICENSE.md
```

If the complete TraceFL-Net-Sim repository is distributed under a separate license, add the corresponding top-level `LICENSE` file and reference it here.
