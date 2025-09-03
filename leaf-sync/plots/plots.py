#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arquivos esperados:
  ../results/stat/stat_metrics_shakespeare_fedavg_c_8_e_1.csv
  ../results/sys/sys_metrics_shakespeare_fedavg_c_8_e_1.csv
"""

import os
import sys

# --- Ajuste do path para conseguir importar visualization_utils.py ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

# Importa as funções fornecidas
from visualization_utils import (
    load_data,
    plot_accuracy_vs_round_number,
    plot_accuracy_vs_round_number_per_client,
    plot_bytes_written_and_read,
    plot_client_computations_vs_round_number,
    get_longest_flops_path,
)

# --- Caminhos dos arquivos conforme solicitado ---
STAT_DIR = os.path.join(CUR_DIR, "..", "results", "stat")
SYS_DIR  = os.path.join(CUR_DIR, "..", "results", "sys")

STAT_FILE = os.path.join(STAT_DIR, "stat_metrics_shakespeare_fedavg_c_8_e_1.csv")
SYS_FILE  = os.path.join(SYS_DIR,  "sys_metrics_shakespeare_fedavg_c_8_e_1.csv")

def _check_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

def main():
    # Garante que os arquivos existem antes de prosseguir
    _check_file(STAT_FILE)
    _check_file(SYS_FILE)

    # Carrega os dataframes usando a função do visualization_utils.py
    stat_metrics, sys_metrics = load_data(
        stat_metrics_file=STAT_FILE,
        sys_metrics_file=SYS_FILE
    )

    # --- Plots principais ---

    # 1) Acurácia média (não ponderada) vs rodada, com barras de desvio padrão
    plot_accuracy_vs_round_number(
        stat_metrics,
        weighted=False,
        plot_stds=True,
        figsize=(10, 6),
        xlabel="Round Number",
        ylabel="Accuracy"
    )

    # 2) Acurácia média ponderada por #amostras vs rodada
    plot_accuracy_vs_round_number(
        stat_metrics,
        weighted=True,
        plot_stds=False,
        figsize=(10, 6),
        xlabel="Round Number",
        ylabel="Accuracy"
    )

    # 3) Acurácia por cliente vs rodada (pontos indicam quando o cliente treinou)
    #    Ajuste max_num_clients conforme o tamanho do seu experimento
    plot_accuracy_vs_round_number_per_client(
        stat_metrics,
        sys_metrics,
        max_num_clients=12,
        figsize=(16, 10)
    )

    # 4) Bytes escritos/lidos pelo servidor (janela móvel)
    plot_bytes_written_and_read(
        sys_metrics,
        rolling_window=10,
        figsize=(10, 6)
    )

    # 5) Computações locais (FLOPs) agregadas por janelas de rounds
    plot_client_computations_vs_round_number(
        sys_metrics,
        aggregate_window=20,
        max_num_clients=20,
        figsize=(20, 10)
    )

    # 6) Caminho de FLOPs mais longo (apenas imprime o valor)
    longest_flops = get_longest_flops_path(sys_metrics)
    print(f"Maior custo de FLOPs (caminho mais longo): {longest_flops}")

if __name__ == "__main__":
    main()
