"""
Solo le figure della sensibilità globale, a partire dai CSV prodotti
da sobol_analysis.py. Nessuna valutazione del modello, nessun indice
ricalcolato: si lancia ogni volta che si cambia la parte cosmetica
(colori, etichette, font, titoli: stanno in
src/cnav/sensitivity_global/plots.py).

Uso:
    python execution/10_sensibilita_globale/sobol_figures.py

Legge:  sobol_aggregate.csv, influence_maps.csv, best_system_map.csv,
        sobol_convergence.csv
Scrive: sobol_aggregate.png, mappe/NN_<gruppo>.png, best_system_map.png,
        sobol_convergence.png

Attenzione a cosa NON si cambia da qui: pesi di aggregazione, gruppi,
inclusione del "others" fra i candidati, NAN_TOL. Quelli cambiano i
numeri delle tabelle e richiedono sobol_analysis.py (che però riusa la
cache delle valutazioni, quindi costa poco)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cnav.sensitivity_global import (
    plot_best_system_map,
    plot_influence_map,
    plot_sobol_aggregate,
    plot_sobol_convergence,
)

FIGURE_DIR = Path(__file__).resolve().parent
MAP_DIR = FIGURE_DIR / "mappe"
DPI = 140

# Baffi del grafico a barre:
#   "range"        dal minimo al massimo di S_T fra tutte le architetture
#                  e tutti i punti della griglia
#   "dispersione"  10°-90° percentile pesato degli stessi valori
#   "bootstrap"    intervallo di confidenza al 95% dell'aggregato
ERROR_BARS = "range"


def _save(ax, path):
    ax.figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(ax.figure)


def make_figures(error_bars: str = ERROR_BARS):
    MAP_DIR.mkdir(exist_ok=True)

    # barre aggregate
    agg = pd.read_csv(FIGURE_DIR / "sobol_aggregate.csv")
    agg.attrs.update(weights=agg["pesi"].iloc[0],
                     spread=(int(agg["spread_lo"].iloc[0]), int(agg["spread_hi"].iloc[0])))
    _save(plot_sobol_aggregate(agg, error=error_bars), FIGURE_DIR / "sobol_aggregate.png")

    # mappe del parametro più influente
    maps = pd.read_csv(FIGURE_DIR / "influence_maps.csv")
    maps["fattibile"] = maps["fattibile"].astype(bool)
    ranges = sorted(maps["range_nmi"].unique())
    speeds = sorted(maps["velocita_kt"].unique())
    for fname, mp in maps.groupby("file", sort=True):
        ax = plot_influence_map(mp, ranges, speeds, mp["gruppo"].iloc[0],
                                weights=mp["pesi"].iloc[0])
        _save(ax, MAP_DIR / f"{fname}.png")

    # controllo: sistema più probabilmente migliore
    bm = pd.read_csv(FIGURE_DIR / "best_system_map.csv")
    bm["fattibile"] = bm["fattibile"].astype(bool)
    _save(plot_best_system_map(bm, ranges, speeds), FIGURE_DIR / "best_system_map.png")

    # convergenza
    conv = pd.read_csv(FIGURE_DIR / "sobol_convergence.csv")
    _save(plot_sobol_convergence(conv), FIGURE_DIR / "sobol_convergence.png")

    print(f"Figure aggiornate in {FIGURE_DIR} ({maps['file'].nunique()} mappe in {MAP_DIR.name}/)")


if __name__ == "__main__":
    make_figures()
