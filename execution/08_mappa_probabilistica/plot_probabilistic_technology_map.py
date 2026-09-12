"""
I grafici a partire dal risultato già calcolato da
execution/07_propagazione/uncertainty_propagation_execution.py.

Non riesegue il modello: legge propagation_result.npz. Cambiare la
soglia di robustezza o la velocità delle bande di confidenza costa
secondi, non ore.

Uso:
    python execution/08_mappa_probabilistica/plot_probabilistic_technology_map.py

Produce:

  1. probabilistic_technology_map.png
     la mappa: colore = tecnologia più probabile, saturazione = quanto.
     La linea tratteggiata nera è il contorno di livello
     max_j P_j = THRESHOLD: dentro, la conclusione è robusta rispetto
     alle incertezze considerate,, fuori è decision-uncertain.

  2. probability_field_<tecnologia>.png
     P per una tecnologia su tutto il piano. Mostra dove una tecnologia
     è competitiva anche quando non vince: informazione che la mappa a
     colori nasconde per costruzione.

  3. intensity_bands_<propulsore>_<velocità>kt.png
     mediana e banda p5-p95 dell'electricity intensity vs range, una
     figura per ciascuna voce di BAND_FIGURES: la versione
     probabilistica della Fig. 4 dell'articolo. L'asse verticale è
     limitato esplicitamente, altrimenti le configurazioni che divergono
     vicino al proprio limite di fattibilità (fuel cell a range corto,
     batteria a range lungo) schiacciano tutto il resto contro l'asse.

  4. boundary_dispersion_<a>__<b>.png  +  boundary_dispersion_all.png
     posizione e dispersione dei confini fra tecnologie (Fase 12,
     punto 7), cioè il risultato su cui la guideline chiede di
     concentrare l'attenzione.

     Le coppie non sono scelte a mano: adjacent_label_pairs() legge
     dalla mappa quali tecnologie si toccano davvero e restituisce solo
     quelle. Con 8 etichette ci sarebbero 28 coppie possibili, ma
     boundary_statistics incrocia due curve senza verificare che una
     delle due sia effettivamente il minimo su tutte: una coppia non
     adiacente produce un "confine" perfettamente calcolabile e
     perfettamente privo di significato, perché sepolto sotto una terza
     tecnologia.
"""
import sys
from itertools import cycle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from cnav.uncertainty import (
    PropagationResult,
    boundary_statistics,
    intensity_percentiles,
    plot_boundary_dispersion,
    plot_intensity_band,
    plot_probability_field,
    plot_probability_map,
    probability_map,
)

RESULT_PATH = Path(__file__).resolve().parents[1] / "07_propagazione" / "propagation_result.npz"
THRESHOLD = 0.90        # soglia di robustezza: la tratteggiata nera sulla mappa
MIN_FRACTION = 0.5      # velocità con meno campioni di così vengono scartate
MIN_CELLS = 3           # celle di contatto minime perché una coppia sia "adiacente"

# Una voce per figura di bande: (velocità richiesta, filtro sulle etichette,
# limiti dell'asse y). La velocità viene approssimata al nodo di griglia più
# vicino: con FlightGrid.default() i nodi sono 180 + 21.33*k, quindi
# 450 -> 457.3 kt e 250 -> 244.0 kt.
BAND_FIGURES = [
    (450.0, "(fan)",       (0.0, 25.0)),
    (250.0, "(propeller)", (0.0, 20.0)),
]

OUT_DIR = Path(__file__).resolve().parent


def adjacent_label_pairs(pmap, min_cells: int = 3):
    """Le coppie di etichette i cui domini si toccano sulla mappa.

    Scorre la mappa del vincitore (argmax_j P_j) e conta, per ogni coppia
    di etichette, quante volte compaiono in due celle contigue lungo il
    range o lungo la velocità. Sono queste le coppie per cui esiste un
    confine vero, ed è su queste che boundary_statistics dice qualcosa.

    min_cells scarta i contatti di una o due celle, che di norma sono
    l'effetto della risoluzione finita della griglia attorno a un punto
    triplo e non un confine esteso.

    Ritorna una lista di (label_a, label_b, n_celle_di_contatto) ordinata
    per lunghezza di contatto decrescente.
    """
    dom = pmap.dominant_index
    counts = {}
    for left, right in ((dom[:, :-1], dom[:, 1:]), (dom[:-1, :], dom[1:, :])):
        differing = left != right
        for a, b in zip(left[differing], right[differing]):
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0) + 1

    out = [(pmap.labels[a], pmap.labels[b], n)
           for (a, b), n in counts.items() if n >= min_cells and a >= 0 and b >= 0]
    return sorted(out, key=lambda t: -t[2])


def _slug(label: str) -> str:
    return (label.lower().replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", ""))


def main():
    result = PropagationResult.load(RESULT_PATH)
    print(f"Caricati {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")

    pmap = probability_map(result)

    # --- 1. la mappa --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_probability_map(pmap, threshold=THRESHOLD, ax=ax,
                         title=f"Tecnologia x propulsore (N = {pmap.n_samples})")
    fig.tight_layout()
    path = OUT_DIR / "probabilistic_technology_map.png"
    fig.savefig(path, dpi=150)
    print(f"  {path.name}")
    plt.close(fig)

    # --- 2. campo di probabilità per ciascuna tecnologia -------------
    for label in pmap.labels:
        if pmap.probability_of(label).max() < 0.01:
            print(f"  (salto {label}: non è mai la migliore in nessun campione)")
            continue
        fig, ax = plt.subplots(figsize=(8.5, 6))
        plot_probability_field(pmap, label, ax=ax)
        fig.tight_layout()
        path = OUT_DIR / f"probability_field_{_slug(label)}.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}")
        plt.close(fig)

    # --- 3. bande di confidenza sull'intensity ------------------------
    for speed_kt, label_filter, ylim in BAND_FIGURES:
        labels = [l for l in result.labels if label_filter in l]
        i = int(np.argmin(np.abs(result.grid.speeds_kt - speed_kt)))
        actual = float(result.grid.speeds_kt[i])

        fig, ax = plt.subplots(figsize=(9.5, 6))
        plot_intensity_band(result, speed_kt=speed_kt, ax=ax, labels=labels)
        ax.set_ylim(*ylim)

        # una curva che esce dall'alto viene tagliata senza avviso: lo si dice
        clipped = []
        for label in labels:
            p50 = intensity_percentiles(result, label, percentiles=(50,))[50][i]
            if np.nanmax(p50) > ylim[1]:
                clipped.append(label.rsplit(" (", 1)[0])
        if clipped:
            ax.text(0.02, 0.97, "fuori scala: " + ", ".join(clipped),
                    transform=ax.transAxes, va="top", fontsize=7.5,
                    style="italic", color="0.35")

        fig.tight_layout()
        path = OUT_DIR / f"intensity_bands_{label_filter.strip('()')}_{actual:.0f}kt.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}  (richiesti {speed_kt:.0f} kt -> nodo {actual:.1f} kt)")
        plt.close(fig)

    # --- 4. dispersione di TUTTI i confini reali ----------------------
    pairs = adjacent_label_pairs(pmap, min_cells=MIN_CELLS)
    if not pairs:
        print("  (nessuna coppia adiacente: la mappa ha un'unica tecnologia dominante)")
        return

    print(f"\nConfini reali trovati sulla mappa ({len(pairs)}):")
    colors = cycle(["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#8172B3", "#937860", "#DA8BC3"])

    fig_all, ax_all = plt.subplots(figsize=(9, 6))
    n_plotted = 0

    for label_a, label_b, n_cells in pairs:
        stats = boundary_statistics(result, label_a, label_b)
        f_max = stats["frazione_con_confine"].max()
        n_ok = int((stats["frazione_con_confine"] >= MIN_FRACTION).sum())
        print(f"  {label_a} / {label_b}")
        print(f"      contatto su {n_cells} celle, frazione_con_confine max = "
              f"{f_max:.2f}, velocità utilizzabili = {n_ok}/{len(stats)}")

        color = next(colors)
        try:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            plot_boundary_dispersion(stats, ax=ax, min_fraction=MIN_FRACTION,
                                     title=f"Confine {label_a} / {label_b}")
            fig.tight_layout()
            path = OUT_DIR / f"boundary_dispersion_{_slug(label_a)}__{_slug(label_b)}.png"
            fig.savefig(path, dpi=150)
            print(f"      -> {path.name}")
            plt.close(fig)
        except ValueError as err:
            print(f"      (nessun grafico: {err})")
            continue

        # stesso confine, sovrapposto agli altri nella figura riassuntiva
        df = stats[stats["frazione_con_confine"] >= MIN_FRACTION].dropna(subset=["p50_nmi"])
        ax_all.fill_betweenx(df["speed_kt"], df["p5_nmi"], df["p95_nmi"],
                             alpha=0.20, color=color, lw=0)
        ax_all.plot(df["p50_nmi"], df["speed_kt"], color=color, lw=2,
                    label=f"{label_a} / {label_b}")
        n_plotted += 1

    if n_plotted:
        ax_all.set_xscale("log")
        ax_all.set_xlabel("Range del confine [nmi]")
        ax_all.set_ylabel("Velocità di crociera [kt]")
        ax_all.set_title(f"Confini fra tecnologie: mediana e banda p5-p95 "
                         f"(N = {result.n_samples})")
        ax_all.legend(fontsize=7, loc="best")
        ax_all.grid(alpha=0.3)
        fig_all.tight_layout()
        path = OUT_DIR / "boundary_dispersion_all.png"
        fig_all.savefig(path, dpi=150)
        print(f"  {path.name}")
    plt.close(fig_all)


if __name__ == "__main__":
    main()