<!-- # Federated-Learning-Network-Workload

Project to Evaluate the Impact of Federated Learning Applications on the Network Workload

## Environment Preparation

### Install virtualenv package

    $ sudo apt install python3-virtualenv

### Create Virtual Environment

    $ virtualenv -p <python-bin> venv
    
* Use ``which`` command to find python3.6 source

### Activate Virtual Environment

    $ source venv/bin/activate

### Install Packages imside Virtual Environment

    $ pip install -r requirements

### -- Desactivate Virtual Environment --

    $ deactivate -->


# TraceFL-Net-Sim

Avaliação do impacto de aplicações de **Federated Learning (FL)** no **tráfego de rede** com realimentação de métricas entre o **LEAF** e um **simulador de rede**.  
O pipeline implementa **FL síncrono** (as rodadas avançam somente após todos os clientes selecionados finalizarem), permitindo estudar a interação entre:

- custo computacional por cliente (FLOPs),
- geração e envio de atualizações de modelo,
- **atrasos, vazão e perdas** na rede,
- e o **desempenho final do modelo**.

<!-- **Figura do pipeline**: veja `docs/pipeline.png` (estágios 1–6). -->

---

## Sumário
- [Arquitetura & Metodologia](#arquitetura--metodologia)
- [Requisitos](#requisitos)
- [Instalação e Ambiente](#instalação-e-ambiente)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Configuração](#configuração)
- [Como Executar (Quickstart)](#como-executar-quickstart)
- [Métricas & Saídas](#métricas--saídas)
- [Reprodutibilidade](#reprodutibilidade)
- [Boas Práticas](#boas-práticas)
- [Licença & Citação](#licença--citação)

---

## Arquitetura & Metodologia

**Estágio 1 — Definição do cenário (LEAF, FL síncrono)**  
- Escolha do **dataset** e **modelo** no LEAF.  
- Definição de **hiperparâmetros** (tamanho do lote, épocas locais, fração de clientes por rodada, etc.).  
- **Tempo de sincronização** habilitado (modo **síncrono**): o servidor só inicia a próxima rodada após **receber todas** as atualizações dos clientes selecionados.

**Estágio 2 — Execução no LEAF & coleta de métricas do sistema**  
- Rodadas de FL no LEAF para **caracterizar workload**: contabilizamos **FLOPs** (ou tempo CPU) por cliente/rodada, tamanho das mensagens (upload/download) e número de amostras processadas.  
- Saída: `leaf_metrics.csv`.

**Estágio 3 — Conversão de FLOPs em tráfego (script Python)**  
- Um **script** transforma **custo computacional (FLOPs)** e **tamanho de atualização** em **tempos de chegada de pacotes** e **taxas de envio**, considerando codecs/compressão (se houver) e MTU.  
- Saída: `traffic_trace.csv` (timestamps, fluxo por cliente, tamanho dos pacotes).

**Estágio 4 — Introdução do tráfego no simulador de rede**  
- O **simulador de rede** recebe `traffic_trace.csv` e reproduz o tráfego FL.  
- Cenários: topologia, capacidade de enlace, filas, latências, perdas e políticas de escalonamento.

**Estágio 5 — Coleta de métricas de rede**  
- Coletamos **atraso**, **jitter**, **vazão** e **perdas** **por cliente e por rodada**.  
- Saída: `network_trace.csv`.

**Estágio 6 — Realimentação no LEAF & métricas do modelo**  
- O **trace de atraso** é injetado no **agendador síncrono** do LEAF: o servidor espera pelos clientes com seus respectivos **delays simulados** antes de agregar.  
- Executamos novamente as rodadas e coletamos **métricas do modelo** (acurácia, loss, etc.).  
- Saída: `model_metrics.csv`.

---

## Requisitos

- **Python**  3.6  
- **virtualenv**
- Dependências Python listadas em `requirements.txt`
- **LEAF** (submódulo ou instalado localmente)
- Simulador de rede (Goland)

---

## Instalação e Ambiente

### 1) Instalar o virtualenv
```bash
sudo apt update && sudo apt install -y python3-virtualenv
```

### 2) Criar ambiente
```bash
virtualenv -p $(which python3) venv
```

### 3) Ativar ambiente
```bash
source venv/bin/activate
```

### 4) Instalar dependências
```bash
pip install -r requirements.txt
```

### 5) (Opcional) Desativar ambiente
```bash
deactivate
```

---

## Estrutura do Repositório

```
TraceFL-Net-Sim/
├─ configs/
│  ├─ example.yaml
├─ docs/
│  └─ pipeline.png
├─ leaf_runner/
│  ├─ run_leaf_baseline.py
│  └─ replay_with_delays.py
├─ leaf-sync/                  # NOVO: FL síncrono (extensão do LEAF)
│  ├─ sync_runner.py           # agendador síncrono
│  └─ utils.py
├─ traffic/
│  └─ flops_to_traffic.py
├─ net_sim/
│  ├─ run_sim.py
│  └─ backends/
│     ├─ ns3.py
│     └─ mininet.py
├─ outputs/
│  ├─ leaf_metrics.csv
│  ├─ traffic_trace.csv
│  ├─ network_trace.csv
│  └─ model_metrics.csv
├─ requirements.txt
└─ README.md
```

---
<!--
## Configuração

Arquivo de exemplo (`configs/example.yaml`):

```yaml
leaf:
  dataset: femnist
  model: cnn
  sync_mode: synchronous
  rounds: 100
  clients_per_round: 0.1
  local_epochs: 1
  batch_size: 20
  lr: 0.01
  seed: 42

traffic:
  mtu_bytes: 1500
  compress_updates: false
  codec: none
  clock_resolution_ms: 1

net_sim:
  backend: ns3
  topology: line-10
  link_capacity_mbps: 100
  link_delay_ms: 10
  queue_discipline: pfifo_fast
  loss_rate: 0.0
  seed: 123

output_dir: outputs
```

---
-->
## Como Executar (Quickstart)

1) Baseline no LEAF:
```bash
python -m leaf_runner.run_leaf_baseline --config configs/example.yaml
```

2) Converter FLOPs → tráfego:
```bash
python -m traffic.flops_to_traffic --leaf outputs/leaf_metrics.csv        --config configs/example.yaml --out outputs/traffic_trace.csv
```

3) Simulação de rede:
```bash
python -m net_sim.run_sim --traffic outputs/traffic_trace.csv        --config configs/example.yaml --out outputs/network_trace.csv
```

4) Reexecução síncrona no LEAF:
```bash
python -m leaf_runner.replay_with_delays --delays outputs/network_trace.csv        --config configs/example.yaml --out outputs/model_metrics.csv
```

---

## Métricas & Saídas

- `leaf_metrics.csv` — workload FL (FLOPs, tempo local, mensagens).  
- `traffic_trace.csv` — tráfego em pacotes.  
- `network_trace.csv` — métricas de rede (delay, jitter, vazão, perdas).  
- `model_metrics.csv` — métricas do modelo após simulação com delays.

---

## Reprodutibilidade

- Fixe `seed` no LEAF e no simulador.  
- Salve trace `.csv` de saída.  
- Registre `seed` do simulador 

---

## Boas Práticas

- Separe workload (LEAF) e rede (simulador).  
- Capture **tamanho real** das mensagens (com cabeçalhos).  
- Relate métricas por rodada e por cliente.

---

## Licença & Citação

```
@software{tracefl_netsim,
  title  = {TraceFL-Net-Sim: Federated Learning Network Workload Simulation with Synchronous Rounds},
  author = {Seu Nome et al.},
  year   = {2025},
  url    = {https://github.com/wocn-unicamp/TraceFL-Net-Sim}
}
```
