def plot_ecdf_multi_full(all_curves, out_path):
    """
    ECDF real, simulada e modelo para múltiplos experimentos,
    com escala log, legendas refinadas e paleta harmônica.
    """
    ensure_dir(os.path.dirname(out_path) or "figures")

    # Paleta harmônica (Set2) – 8 cores suaves
    colors = plt.get_cmap("Set2").colors

    # Figura mais larga para dar espaço às legendas
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)

    color_handles = []
    seen_labels = set()

    # Mapeia nomes simplificados
    name_map = {
        "fedavg_c_50_e_1": "Batch 100%",
        "minibatch_c_20_mb_0.9": "Batch 90%",
        "minibatch_c_20_mb_0.8": "Batch 80%",
        "minibatch_c_20_mb_0.6": "Batch 60%",
        "minibatch_c_20_mb_0.5": "Batch 50%",
        "minibatch_c_20_mb_0.4": "Batch 40%",
        "minibatch_c_20_mb_0.2": "Batch 20%",
    }

    # --- Curvas principais ---
    for i, cur in enumerate(all_curves):
        lbl = name_map.get(cur["label"], cur["label"])
        color = colors[i % len(colors)]

        ax.step(cur["x_real"], cur["y_real"], where="post",
                lw=2.0, color=color, alpha=0.9)
        ax.step(cur["x_sim"], cur["y_sim"], where="post",
                lw=1.6, ls="--", color=color, alpha=0.9)
        ax.plot(cur["x_model"], cur["y_model"],
                lw=1.8, ls=":", color=color, alpha=0.9)

        if lbl not in seen_labels:
            color_handles.append(Line2D([0], [0], color=color, lw=2.8, label=lbl))
            seen_labels.add(lbl)

    # --- Eixos e estilo ---
    # ax.set_xscale("log")
    ax.set_xlabel("GFLOPs", fontsize=15, labelpad=6)
    ax.set_ylabel("Clients (%)", fontsize=15, labelpad=6)
    ax.tick_params(axis="both", labelsize=13, length=5, width=1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _style_xaxis(ax)

    # --- Legenda 1: tipo de curva (estilos de linha) ---
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Real (ECDF)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Sim (ECDF)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle=":", label="Model (CDF)"),
    ]
    leg_style = ax.legend(
        handles=style_handles,
        title="Curve Type",
        title_fontsize=13,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.5),  # fora do quadro, empilhada
        frameon=True,
        fancybox=True,
        edgecolor="0.6",
        ncol=1,
    )
    ax.add_artist(leg_style)

    # --- Legenda 2: cores / configuração ---
    leg_colors = ax.legend(
        handles=color_handles,
        title="Configuration",
        title_fontsize=13,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.0),
        frameon=True,
        fancybox=True,
        edgecolor="0.6",
    )
    fig.add_artist(leg_colors)

    # --- Margens para espaço confortável ---
    plt.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.12)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)