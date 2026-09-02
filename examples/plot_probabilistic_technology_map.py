"""
I grafici a partire dal risultato già calcolato da
examples/uncertainty_propagation_execution.py.

Non riesegue il modello: legge propagation_result.npz. Cambiare la
soglia di robustezza o la velocità delle bande di confidenza costa
secondi, non ore.

Uso:
    python examples/plot_probabilistic_technology_map.py

Produce quattro figure, che sono i quattro modi diversi di guardare lo
stesso risultato:

  1. probabilistic_technology_map.png
     la mappa: colore = tecnologia più probabile, saturazione = quanto.
     Sarebbe il "risultato finale ideale", la generalizzazione
     probabilistica della mappa deterministica

  2. probability_field_<tecnologia>.png
     P per una tecnologia su tutto il piano. Mostra dove una tecnologia
     è competitiva anche quando non vince: informazione che la mappa a
     colori nasconde per costruzione

  3. intensity_bands.png
     mediana e banda p5-p95 dell'electricity intensity vs range: la
     versione probabilistica della Fig. 4 dell'articolo

  4. boundary_dispersion.png
     posizione e dispersione di un confine fra tecnologie (Fase 12,
     punto 7), cioè il risultato su cui la guideline chiede di
     concentrare l'attenzione
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from cnav.uncertainty import (
    PropagationResult,
    aggregate_to_technologies,
    boundary_statistics,
    plot_boundary_dispersion,
    plot_intensity_band,
    plot_probability_field,
    plot_probability_map,
    probability_map,
)

RESULT_PATH = "propagation_result.npz"
THRESHOLD = 0.90
BAND_SPEED_KT = 450.0
BOUNDARY = ("Hydrogen fuel cell (propeller)", "e-SAF combustion (fan)")

OUT_DIR = Path(__file__).resolve().parent


def main():
    result = PropagationResult.load(RESULT_PATH)
    print(f"Caricati {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")

    pmap = probability_map(result)
    agg = aggregate_to_technologies(pmap)

    # --- 1. la mappa (entrambe le versioni, affiancate) ---------------
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    plot_probability_map(pmap, threshold=THRESHOLD, ax=axes[0],
                         title=f"Tecnologia x propulsore (N = {pmap.n_samples})")
    plot_probability_map(agg, threshold=THRESHOLD, ax=axes[1],
                         title=f"Tecnologia aggregata (N = {agg.n_samples})")
    fig.tight_layout()
    path = OUT_DIR / "probabilistic_technology_map.png"
    fig.savefig(path, dpi=150)
    print(f"  {path.name}")
    plt.close(fig)

    # --- 2. campo di probabilita' per ciascuna tecnologia -------------
    for label in agg.labels:
        if agg.probability_of(label).max() < 0.01:
            print(f"  (salto {label}: non vince mai)")
            continue
        fig, ax = plt.subplots(figsize=(8.5, 6))
        plot_probability_field(agg, label, ax=ax)
        fig.tight_layout()
        slug = label.lower().replace(" ", "_").replace("-", "_")
        path = OUT_DIR / f"probability_field_{slug}.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}")
        plt.close(fig)

    # --- 3. bande di confidenza sull'intensity ------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6))
    plot_intensity_band(result, speed_kt=BAND_SPEED_KT, ax=ax,
                        labels=[l for l in result.labels if "(fan)" in l])
    fig.tight_layout()
    path = OUT_DIR / "intensity_bands.png"
    fig.savefig(path, dpi=150)
    print(f"  {path.name}")
    plt.close(fig)

    # --- 4. dispersione di un confine ---------------------------------
    stats = boundary_statistics(result, *BOUNDARY)
    try:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_boundary_dispersion(stats, ax=ax,
                                 title=f"Confine {BOUNDARY[0]} / {BOUNDARY[1]}")
        fig.tight_layout()
        path = OUT_DIR / "boundary_dispersion.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}")
        plt.close(fig)
    except ValueError as err:
        print(f"  (nessun grafico del confine: {err})")


if __name__ == "__main__":
    main()
