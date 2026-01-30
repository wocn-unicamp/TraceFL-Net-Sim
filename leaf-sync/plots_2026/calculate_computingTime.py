import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

DATA_DIR="../results/sys/"; OUT_DIR="trace_sim"; FIG_DIR="figures/cdf_computingTime"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

C_FIXED=64; E_VALUES=[1,2,3,4,5,6]; DEVICE_FLOPS_PER_SEC=64e9

def plot_cdf_single(x, title, out_path):
    x=np.sort(np.asarray(x)); y=np.arange(1,len(x)+1)/len(x)
    plt.figure(figsize=(7,5)); plt.plot(x,y,linewidth=2)
    plt.title(title); plt.xlabel("Computing Time (s)"); plt.ylabel("% Clients that finished training")
    plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close()

def plot_cdf_group(cdf_map, title, out_path):
    plt.figure(figsize=(7,5))
    for label,x in cdf_map.items():
        x=np.sort(np.asarray(x)); y=np.arange(1,len(x)+1)/len(x)
        plt.plot(x,y,linewidth=2,label=label)
    plt.title(title); plt.xlabel("Computing Time (s)"); plt.ylabel("% Clients that finished training")
    plt.grid(True,alpha=0.3)
    plt.legend(loc="lower right")  # evita loc="best" (lento)
    plt.tight_layout()
    plt.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close()

cdf_by_e={}
for e in E_VALUES:
    in_path=os.path.join(DATA_DIR,f"sys_metrics_fedavg_c_{C_FIXED}_e_{e}.csv")
    if not os.path.exists(in_path): print(f"[warn] No existe: {in_path}"); continue

    df=pd.read_csv(in_path,header=None,names=["client_id","round","aux","num_samples","phase","bytes_up","bytes_down","flops"])
    df["flops"]=pd.to_numeric(df["flops"],errors="coerce")
    df=df.dropna(subset=["flops"]).copy()
    df["computingTime"]=df["flops"]/DEVICE_FLOPS_PER_SEC

    out_csv=os.path.join(OUT_DIR,f"sys_metrics_fedavg_c_{C_FIXED}_e_{e}_with_computingTime.csv")
    df.to_csv(out_csv,index=False); print(f"[ok] Guardado CSV: {out_csv}")

    x=df["computingTime"].dropna().to_numpy()
    cdf_by_e[f"e={e}"]=x

    out_fig=os.path.join(FIG_DIR,f"cdf_computingTime_c_{C_FIXED}_e_{e}.png")
    plot_cdf_single(x,f"Cumulative Distribution Function of % Clients with c={C_FIXED}",out_fig)
    print(f"[ok] Guardada figura: {out_fig}")

if cdf_by_e:
    out_fig=os.path.join(FIG_DIR,f"cdf_computingTime_c_{C_FIXED}_ALL_e.png")
    plot_cdf_group(cdf_by_e,f"Cumulative Distribution Function of % Clients with c={C_FIXED}",out_fig)
    print(f"[ok] Guardada figura (grupo): {out_fig}")
