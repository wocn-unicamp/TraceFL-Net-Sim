# TraceFL-Net-Sim

**TraceFL-Net-Sim** is a trace-driven discrete-event simulator for evaluating how Federated Learning (FL) workloads interact with background traffic in bandwidth-constrained access networks.

The project combines three components:

1. a modified synchronous FL workflow based on LEAF;
2. a trace-processing pipeline that converts computational demand into client completion times;
3. a Go-based discrete-event network simulator.

This repository accompanies the manuscript:

> **Probabilistic Modeling and Performance Analysis of Federated Learning Traffic over Access Networks**

---

## Overview

The experimental workflow is:

```text
FL experiments
  leaf-sync/
      |
      | per-client system metrics
      | FLOP/round, model-update size, client, round
      v
Trace preparation
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

The FL and network stages are deliberately separated. FL execution produces the computational workload, while TraceFL-Net-Sim determines how the corresponding model updates are affected by the communication network.

The evaluated FL setting is synchronous and round-based. The central server waits for all participating clients before completing a training round.

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
├── leaf-sync/                    # Synchronous FL workload generation
├── traces/                       # Trace conversion, validation, aggregation, plots
├── trace_driven_simulator/       # Go discrete-event network simulator
├── params/                       # Network-simulation parameter files
├── speed_up_eval/                # Complementary computational analysis
└── figures/                      # Additional project/manuscript figures
```

### Main subdirectories

```text
leaf-sync/
├── baseline/
├── benchmark_comp_capacity/
├── data/
├── datasets/
├── docs/
├── generador_trafico/
├── models/
├── paper_experiments/
├── plots/
├── plots_2026/
├── results/
├── LICENSE.md
├── README.md
└── requirements.txt
```

```text
traces/
├── convert_flops_to_trace.py
├── join_traces.py
├── plot_bimodal_traces.py
├── plot_cdf_computation_time.py
├── plot_fl_training_time.py
├── plot_verificar_traces.py
├── sys/
├── sys_bimodal/
├── sys_bimodal_amdahl/
├── net/
├── net_join/
├── stat/
├── normalize/
└── figures/
```

```text
trace_driven_simulator/
├── main.go
├── data_processor.py
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

---

# 1. FL Workload Generation

The first stage is performed under:

```text
leaf-sync/
```

This directory contains the modified LEAF-based environment used to execute the FL applications and collect the workload information required by the network simulator.

The experiments generate system metrics for each client and training round, including:

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

Available scripts include:

```text
femnist.sh
shakespeare.sh
sent140.sh
test.sh
```

The current study primarily uses:

- **FEMNIST with a CNN**;
- **Shakespeare with an LSTM**.

Additional analysis utilities are available in:

```text
leaf-sync/plots_2026/
```

including scripts for model accuracy, computational demand, computation time, target accuracy, and total computation time.

## Running an FL experiment

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

The required system-metrics traces are subsequently placed under:

```text
traces/sys/
```

---

# 2. Original FL Traces

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

This script generates grouped CDFs under:

```text
traces/figures/grouped_cdfs/
```

These plots are used as consistency checks when comparing traces obtained from different FL configurations.

---

# 3. FLOP-to-Trace Conversion

The main conversion script is:

```text
traces/convert_flops_to_trace.py
```

Its purpose is to convert the per-client computational demand reported by LEAF into the local computation times used to determine when model updates become available to the network simulator.

For client \(i\) in round \(r\),

\[
T_{i,r}
=
\frac{X_{i,r}}
{C^{\mathrm{eff}}_{i,d}\times 10^9},
\]

where:

- \(X_{i,r}\) is the computational demand in `FLOP/round`;
- \(C^{\mathrm{eff}}_{i,d}\) is the effective processing capacity in GFLOP/s.

The script preserves the original eight LEAF fields and adds:

| Position | Field | Description |
|---:|---|---|
| 9 | `capacity_gflops` | effective processing capacity assigned to the client |
| 10 | `time` | local computation time in seconds |

The original traces in `traces/sys/` are not modified.

## Heterogeneous processing capacities

The client base capacities follow a bimodal distribution:

```text
Mode 1: N(0.5, 0.12²) GFLOP/s
Mode 2: N(1.5, 0.12²) GFLOP/s
```

with base capacities restricted to:

```text
[0.20, 1.80] GFLOP/s
```

Clients are divided between the two capacity modes and each client receives a fixed processing capacity that is maintained across training rounds.

## Amdahl scaling

The current journal configuration enables Amdahl-based scaling.

For application \(d\),

\[
S_d =
\frac{1}
{(1-P_d)+P_d/N_c},
\]

and

\[
C^{\mathrm{eff}}_{i,d}
=
C^{\mathrm{base}}_i S_d.
\]

The configured values are:

| Application | \(P_d\) | Cores | Speedup |
|---|---:|---:|---:|
| FEMNIST/CNN | 0.95 | 4 | 3.478 |
| Shakespeare/LSTM | 0.40 | 4 | 1.429 |

Run the conversion with:

```bash
cd traces
python convert_flops_to_trace.py
```

With Amdahl scaling enabled, the generated traces are stored in:

```text
traces/sys_bimodal_amdahl/
```

The script also generates client-capacity mappings and capacity-distribution plots.

---

# 4. Validation of Converted Traces

The converted computation-time traces can be inspected using:

```text
traces/plot_bimodal_traces.py
```

Run:

```bash
python plot_bimodal_traces.py
```

When analyzing the Amdahl-enabled traces, configure the script to read:

```text
sys_bimodal_amdahl/
```

rather than `sys_bimodal/`.

The generated CDFs are stored under:

```text
traces/figures/bimodal_cdfs/
```

or:

```text
traces/figures/bimodal_cdfs_amdahl/
```

The bimodal model applies to **processing capacity**. The resulting computation-time distribution is not expected to be bimodal because computation time depends jointly on both computational demand and client capacity.

---

# 5. Network Simulator

The discrete-event simulator is implemented in Go under:

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

- `main.go`: simulator entry point;
- `internal/simulator/`: simulation models, event-processing functions, and queues;
- `packages/writer/`: output models and routines for writing simulation metrics;
- `data_processor.py`: auxiliary trace-processing utility associated with the simulator workflow.

TraceFL-Net-Sim models Ethernet-frame transmission from FL clients and background-traffic sources toward a central server.

The network model includes:

- client access links;
- traffic-source queues;
- a shared output link;
- FCFS service;
- configurable link capacities;
- propagation delay;
- probabilistic frame losses;
- retransmissions;
- frame delivery to the central server.

The simulator is technology-agnostic. It models contention and queueing in a shared access network rather than the MAC scheduling or bandwidth-allocation mechanism of a specific 5G, Wi-Fi, or PON implementation.

---

# 6. Background Traffic

The simulator supports three concurrent background-traffic profiles:

| Traffic profile | Representative service | Temporal behavior |
|---|---|---|
| Poisson | Web-like traffic | exponential inter-arrival times |
| Pareto | multimedia-like traffic | bursty / heavy-tailed behavior |
| CBR | VoIP-like traffic | periodic frame generation |

The background traffic competes with FL traffic for the shared network resources.

---

# 7. Running the Network Simulation

Network experiments are launched from the repository root using:

```text
run_simulation.sh
```

Run:

```bash
./run_simulation.sh
```

Simulation parameter files are maintained under:

```text
params/
```

The converted FL traces are used as the FL workload input to the Go simulator.

The network simulator produces one output trace for each configured random seed. These files are stored in:

```text
traces/net/
```

The simulation outputs contain the network-level information used in the subsequent analysis, including client/round information, computation time, and queueing/communication measurements.

---

# 8. Combining Simulation Seeds

The Go simulator produces a separate output for each seed.

The script

```text
traces/join_traces.py
```

combines outputs corresponding to the same experiment.

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

The seed suffix is removed from the resulting experiment name, while observations from all seeds are preserved.

---

# 9. Network and Training-Time Analysis

Several scripts under `traces/` operate on the network-simulation outputs.

## Computation-time CDFs

```text
plot_cdf_computation_time.py
```

reads the joined network traces and generates grouped CDFs of the `computation-time` field.

Run:

```bash
python plot_cdf_computation_time.py
```

Output:

```text
traces/figures/net_grouped_cdfs/
```

## FL training time

```text
plot_fl_training_time.py
```

analyzes the training duration obtained after incorporating the simulated network communication delays.

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

# 10. Trace Directory Summary

The main data directories under `traces/` are:

| Directory | Purpose |
|---|---|
| `sys/` | original FL system traces |
| `sys_bimodal/` | traces with heterogeneous capacities without Amdahl scaling |
| `sys_bimodal_amdahl/` | traces with heterogeneous capacities and Amdahl scaling |
| `net/` | network-simulator outputs separated by seed |
| `net_join/` | network outputs combined across seeds |
| `stat/` | auxiliary statistical data used by the analysis workflow |
| `normalize/` | auxiliary normalized data used by processing/analysis scripts |
| `figures/` | validation and result figures |

The `traces/` directory contains its own README with additional implementation details.

---

# 11. Complementary Computational Analysis

The directory

```text
speed_up_eval/
```

contains complementary computational analyses used to study processing-capacity effects outside the detailed discrete-event network simulation.

These analyses support the evaluation of quantities such as:

- client computation time;
- round duration;
- computational heterogeneity;
- offered FL network load.

---

# Installation

The repository contains Python and Go components.

## Python

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the root dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The LEAF-based component also contains:

```text
leaf-sync/requirements.txt
```

Because the LEAF code may depend on an older Python/software stack, a separate environment may be preferable for `leaf-sync/`.

## Go

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

# Typical End-to-End Execution

A typical experiment follows the sequence below.

## Step 1 — Run the FL application

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

## Step 2 — Check the original computational-demand traces

From the repository root:

```bash
cd traces
python plot_verificar_traces.py
```

## Step 3 — Convert FLOP/round into computation time

```bash
python convert_flops_to_trace.py
```

For the current journal configuration, the processed traces are written to:

```text
sys_bimodal_amdahl/
```

## Step 4 — Check the generated computation-time traces

Configure `plot_bimodal_traces.py` to use `sys_bimodal_amdahl/`, then run:

```bash
python plot_bimodal_traces.py
```

## Step 5 — Run TraceFL-Net-Sim

Return to the repository root:

```bash
cd ..
./run_simulation.sh
```

Per-seed network outputs are written under:

```text
traces/net/
```

## Step 6 — Join the network outputs

```bash
cd traces
python join_traces.py
```

Combined traces are written to:

```text
traces/net_join/
```

## Step 7 — Generate final analyses

```bash
python plot_cdf_computation_time.py
python plot_fl_training_time.py
```

The generated figures are available under:

```text
traces/figures/
```

---

# Modeling Scope

TraceFL-Net-Sim is intended to evaluate the interaction between FL workload characteristics and communication-network constraints.

The current model assumes:

- synchronous, round-based FL;
- fixed client processing capacity across training rounds;
- computation time derived from per-round computational demand;
- heterogeneous client processing capacities;
- FCFS queueing;
- configurable background traffic;
- configurable frame-loss probability and retransmission;
- a shared access-network bottleneck.

Technology-specific MAC scheduling and bandwidth-allocation mechanisms are outside the current abstraction. Therefore, the reported delays characterize the configured shared access-network scenario rather than a specific access technology.

---

# Citation

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

# License

The modified LEAF component includes its license under:

```text
leaf-sync/LICENSE.md
```

Add the corresponding top-level project license here if the complete TraceFL-Net-Sim repository is distributed under a separate license.
