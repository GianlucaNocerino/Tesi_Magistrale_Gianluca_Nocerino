"""
Screening di Morris.

Presuppone theta_acc.csv e non ha bisogno del risultato della
propagazione: è un'analisi indipendente.

Uso:
    python examples/morris_screening.py

L'output è uno solo, la renewable electricity intensity, valutata però
in più punti del piano range-velocità: Morris ha bisogno di uno
scalare per campione, e la intensity è una superficie. I punti sono le
missioni rappresentative di ciascun sistema propulsivo

Produce:
  morris_screening.csv        tabella tidy (output x fattore: mu, mu*, sigma, n_ee)
  morris_cloud.csv            le stesse righe con coordinate normalizzate,
                              distanza dall'origine e flag di selezione
  morris_clouds.png           una nube (mu*, sigma) per punto operativo
  morris_cloud_cumulativa.png la nube media su tutti i punti

Da qui si sceglie manualmente il sottoinsieme di fattori per la Sobol: lo
screening ordina, non decide. Tre cose da guardare, non solo mu*:

  - distanza dall'origine sulla nube = influenza complessiva. Il filtro
    è sulla distanza e non sul solo mu* perchè un fattore con sigma
    alto va tenuto anche a mu* medio: è li' che S_Ti - S_i sarà
    diverso da zero
  - un fattore che domina un solo punto operativo (colonna
    n_selezionato della nube cumulativa) va tenuto anche se in media
    conta poco: è proprio il caso della batteria, che vive in un
    angolo minuscolo del piano
  - n_ee molto minore di r = il fattore ha spesso portato Y fuori dal
    dominio di fattibilità. Statistiche stimate su pochi passi:
    sospetto, non "poco influente"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from cnav.uncertainty import load_theta_acc
from cnav.sensitivity_global import (
    FactorSpace,
    aggregate_morris_cloud,
    default_outputs,
    evaluate_outputs,
    morris_cloud_table,
    morris_indices,
    morris_trajectories,
    nan_report,
    outputs_table,
    plot_aggregate_morris_cloud,
    plot_all_morris_clouds,
    ranking_table,
    select_factors,
)

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
THETA_ACC_PATH = "theta_acc.csv"
N_TRAJECTORIES = 20        # r: 10 per provare, 20-30 per il risultato da tesi
N_LEVELS = 8               # p: griglia dei livelli, pari
SEED = 0
OUTPUT_CSV = "morris_screening.csv"
CLOUD_CSV = "morris_cloud.csv"
FIGURE_DIR = Path(__file__).resolve().parent

# Soglia dei grafici a nube: frazione della distanza massima
# dall'origine, per punto operativo. Sarebbe una soglia di lettura, non un
# test: si guarda il grafico, la si cambia, si riguarda. Per esplorarla
# a mano c'è plot_morris_cloud_interactive (slider), che però vuole un
# backend interattivo
CLOUD_THRESHOLD = 0.10

# I punti operativi. default_outputs() = la intensity di ogni sistema
# nelle sue missioni rappresentative. Per cambiarli si modifica
# REPRESENTATIVE_MISSIONS in cnav/sensitivity_global/outputs.py, oppure
# si passa qui outputs_from_missions({...})
OUTPUTS = default_outputs()


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)

    theta_acc = load_theta_acc(THETA_ACC_PATH)
    space = FactorSpace.default(theta_acc)

    print("=" * 78)
    print("SPAZIO DEI FATTORI")
    print("=" * 78)
    print(f"{space.k} fattori (quantili in [0,1], non valori fisici):")
    for i, nm in enumerate(space.names):
        print(f"  {i:2d}  {nm}")
    print(f"\nI 3 parametri derivati e i {len(theta_acc.columns)} di calibrazione non sono")
    print("fattori: i primi sono funzioni deterministiche dentro il modello,")
    print("i secondi entrano in blocco come 'calibration_block'.")

    print("\n" + "=" * 78)
    print("PUNTI OPERATIVI (output = renewable electricity intensity)")
    print("=" * 78)
    print(outputs_table(OUTPUTS).to_string(index=False))

    # -----------------------------------------------------------------
    # Disegno
    # -----------------------------------------------------------------
    U, moved = morris_trajectories(space.k, r=N_TRAJECTORIES, p=N_LEVELS, seed=SEED)
    theta = space.theta_from_unit(U)

    print("\n" + "=" * 78)
    print("SCREENING DI MORRIS")
    print("=" * 78)
    print(f"{N_TRAJECTORIES} traiettorie x ({space.k} + 1) = {len(U)} valutazioni, "
          f"{len(OUTPUTS)} punti operativi per valutazione")

    t0 = time.time()
    Y = evaluate_outputs(theta, OUTPUTS, verbose=True, progress_every=50)
    dt = time.time() - t0
    print(f"Tempo: {dt:.0f}s ({dt / len(U):.2f}s per valutazione)")

    # -----------------------------------------------------------------
    # NaN prima di tutto
    # -----------------------------------------------------------------
    print("\nOutput non definiti (guardare questa tabella prima degli indici):")
    print(nan_report(Y).to_string())
    print("\n  Frazione alta = il punto operativo cade spesso fuori dal dominio")
    print("  di fattibilità del sistema. Va spostato più corto o più lento")
    print("  in REPRESENTATIVE_MISSIONS, non interpretato")

    # -----------------------------------------------------------------
    # Indici
    # -----------------------------------------------------------------
    df = morris_indices(U, moved, Y, space.names)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTabella completa salvata in {OUTPUT_CSV}")

    print("\n" + "=" * 78)
    print("RANKING (mu* normalizzato per colonna, ordinato per media)")
    print("=" * 78)
    print(ranking_table(df).round(3).to_string())

    print("\n" + "=" * 78)
    print("EFFETTI CALCOLABILI (n_ee su r)")
    print("=" * 78)
    print(df.pivot(index="fattore", columns="output", values="n_ee").to_string())

    # -----------------------------------------------------------------
    # Nubi
    # -----------------------------------------------------------------
    cloud = morris_cloud_table(df, threshold=CLOUD_THRESHOLD)
    cloud.to_csv(CLOUD_CSV, index=False)

    fig = plot_all_morris_clouds(df, threshold=CLOUD_THRESHOLD, ncols=3)
    clouds_path = FIGURE_DIR / "morris_clouds.png"
    fig.savefig(clouds_path, dpi=140)

    ax = plot_aggregate_morris_cloud(df, threshold=CLOUD_THRESHOLD)
    ax.legend(fontsize=8, loc="upper left")
    cumulative_path = FIGURE_DIR / "morris_cloud_cumulativa.png"
    ax.figure.savefig(cumulative_path, dpi=140)

    agg = aggregate_morris_cloud(df, threshold=CLOUD_THRESHOLD)
    print("\n" + "=" * 78)
    print(f"NUBE CUMULATIVA (soglia = {CLOUD_THRESHOLD:.0%} della distanza massima)")
    print("=" * 78)
    print(agg.round(3).to_string(index=False))

    proposta = select_factors(df, threshold=CLOUD_THRESHOLD, dominanza=0.5)
    print(f"\nProposta per la Sobol ({len(proposta)} fattori su {space.k}):")
    for nm in proposta:
        print(f"  - {nm}")
    print("\nè una proposta: la scelta definitiva si fa a mano guardando le")
    print("nubi, perchè sulla Sobol pesa direttamente il costo N*(k+2).")
    print(f"Figure salvate in {FIGURE_DIR}:")
    print(f"  {clouds_path.name}")
    print(f"  {cumulative_path.name}")

    plt.close("all")


if __name__ == "__main__":
    main()
