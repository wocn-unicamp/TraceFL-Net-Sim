# TraceFL-Net-Sim

**TraceFL-Net-Sim** is a trace-driven discrete-event simulator for studying the interaction between Federated Learning (FL) traffic and background traffic in bandwidth-constrained access networks.

This repository accompanies the manuscript:

> **Probabilistic Modeling and Performance Analysis of Federated Learning Traffic over Access Networks**

## Overview

TraceFL-Net-Sim combines FL workload traces with a discrete-event network simulator implemented in Go. Per-client computational demand is converted into local computation times, which determine when model updates become available for transmission. The resulting traffic is then evaluated under configurable access-network conditions and coexisting background traffic.

The evaluated FL workflow is synchronous and round-based: a round is completed only after the model updates from all participating clients reach the central server.

## Main Features

- Trace-driven FL traffic simulation
- Per-client and per-round computational-demand traces
- Heterogeneous client processing capacities
- Amdahl-based processing-capacity scaling
- Poisson, Pareto, and CBR background traffic
- Ethernet frame-level discrete-event simulation
- FCFS queueing
- Configurable link capacities and propagation delays
- Probabilistic frame loss and retransmission
- Multiple independent simulation seeds
- Per-client, per-round, and queueing metrics

## Repository Structure

```text
TraceFL-Net-Sim/
├── trace_driven_simulator/   # Core Go discrete-event simulator
├── leaf-sync/                # Synchronous FL workload generation
├── traces/                   # Trace preparation, processing, aggregation, and plots
├── speed_up_eval/            # Complementary analysis
├── run_simulation.sh         # Simulation workflow
├── requirements.txt          # Python dependencies
├── go.mod
├── go.sum
└── README.md
```

The `traces/` directory contains its own documentation describing the complete trace-processing workflow.

## Trace Processing Pipeline

The trace-processing workflow is organized under `traces/`:

```text
LEAF traces (sys/)
        |
        v
processing / validation
        |
        v
computational-capacity assignment
and computation-time generation
        |
        v
network simulator
        |
        v
per-seed network traces (net/)
        |
        v
aggregation across seeds (net_join/)
        |
        v
analysis and figures
```

The main directories are:

| Directory | Purpose |
|---|---|
| `traces/sys/` | Original FL system traces used as input |
| `traces/sys_gen/` | Regenerated traces used for validation before replacing input traces |
| `traces/sys_bimodal/` | Traces extended with client processing capacity and computation time |
| `traces/sys_bimodal_amdahl/` | Traces generated with Amdahl-based effective capacities |
| `traces/net/` | Network-simulator output for individual seeds |
| `traces/net_join/` | Network traces aggregated across simulation seeds |
| `traces/figures/` | Figures generated from trace and network results |

See [`traces/README.md`](traces/README.md) for the detailed workflow, script descriptions, and trace formats.

## Computational Heterogeneity

Client processing capacities are modeled using a bimodal distribution. Each client is assigned a fixed base processing capacity that is maintained across training rounds.

For client \(i\) in round \(r\),

\[
T_{i,r} =
\frac{X_{i,r}}
{C^{\mathrm{eff}}_{i,d} \times 10^9},
\]

where \(X_{i,r}\) is the computational demand in FLOP/round and \(C^{\mathrm{eff}}_{i,d}\) is the effective processing capacity in GFLOP/s.

For the journal experiments, effective processing capacity is obtained using Amdahl's law with four processing cores and application-specific parallelizable fractions.

Trace generation and capacity assignment are implemented in:

```text
traces/generate_bimodal_traces.py
```

Detailed parameters and output columns are documented in `traces/README.md`.

## Requirements

The project requires:

- Go
- Python
- Python dependencies listed in `requirements.txt`

Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Go dependencies:

```bash
go mod download
```

## Running the Experiments

The main simulation workflow is executed from the repository root using:

```bash
./run_simulation.sh
```

Trace preprocessing and analysis scripts are executed from:

```text
traces/
```

For example:

```bash
cd traces
python generate_bimodal_traces.py
python join_traces.py
```

Before reproducing the journal experiments, verify the parameter settings documented in `traces/README.md`, including the Amdahl configuration and random seeds.

## Background Traffic

TraceFL-Net-Sim supports concurrent background traffic with distinct temporal characteristics:

| Profile | Representative traffic |
|---|---|
| Poisson | Web-like traffic |
| Pareto | Bursty multimedia traffic |
| CBR | VoIP-like traffic |

These traffic sources compete with FL traffic for the shared network resources.

## Reproducibility

The trace-processing scripts use deterministic seeds where applicable. Network experiments are executed with multiple independent seeds, and the resulting traces can be combined using:

```text
traces/join_traces.py
```

The exact code version associated with a submitted manuscript should be preserved with a Git tag or release.

Example:

```bash
git tag -a jisa-r1 -m "Artifact for JISA revision R1"
git push origin jisa-r1
```

For reproducibility, the repository should preserve:

- the FL input traces;
- the scripts used to generate computation times;
- the network-simulation parameters;
- the random seeds;
- the code version associated with the reported results.

## Scope

TraceFL-Net-Sim provides a technology-agnostic abstraction of a shared access network. Contention is represented through configured link capacities and FCFS queues.

The simulator does not reproduce technology-specific MAC-layer scheduling or bandwidth-allocation mechanisms used by systems such as 5G, Wi-Fi, or PON. Consequently, the measured delays characterize the modeled bandwidth-constrained access-network scenario and should not be interpreted as direct predictions for a specific access technology.

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

## License

See the `LICENSE` file for licensing information.
