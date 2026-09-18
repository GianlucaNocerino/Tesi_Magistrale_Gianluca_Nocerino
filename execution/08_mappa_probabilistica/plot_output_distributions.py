"""
Le distribuzioni dell'output nei punti operativi.

Legge propagation_result.npz (prodotto dalla Fase 7) e ne estrae, per
alcuni punti operativi scelti, la PDF della renewable electricity
intensity di ciascun sistema propulsivo. Non riesegue la propagazione.

Uso:
    python execution/08_mappa_probabilistica/plot_output_distributions.py

Produce, per ogni voce di PUNTI:

  1. output_pdf_<punto>.png
     le PDF dei quattro sistemi con quel propulsore, sovrapposte.
     Ogni curva è moltiplicata per P(fattibile), quindi l'area sotto la
     curva è la probabilità che quella configurazione esista: un
     sistema fattibile in un terzo dei campioni non deve sembrare alla
     pari con uno fattibile sempre. In legenda P(fatt) e P(migliore)

  2. output_diff_<punto>.png
     la PDF della differenza fra i due sistemi più probabili nel punto,
     valutati sullo stesso theta. Sarebbe il grafico che dice davvero se 
     il confronto è deciso: due marginali sovrapposte non significano
     confronto incerto, perchè i due sistemi condividono i parametri e
     si spostano insieme

  3. output_points_<sistema>.png
     lo stesso sistema in tutti i punti, come violini affiancati.
     La sagoma è calcolata fra i percentili CLIP_VIOLINI, mediana e
     banda sul campione completo: il numero di campioni lasciati fuori
     dall'asse è scritto sulla figura

  4. output_distributions.csv
     la tabella dei numeri (P_fattibile, P_migliore, media, sd, cv,
     percentili) per ogni punto e ogni sistema

MODO ESATTO
-----------
Con RIVALUTA = True i punti non vengono approssimati al nodo di griglia
più vicino, ma rivalutati esattamente, riusando la theta salvata dentro
il .npz: stessi campioni della mappa probabilistica, punto esatto. Serve
quando i punti sono quelli di MISSION_LIBRARY usati nell'analisi di
sensibilità (100 / 1500 / 6000 nmi) e si vuole che le due analisi
parlino dello stesso punto: la griglia di default è geometrica con
passo del 17%, quindi lo snap può spostare il range dell'8%.
Costa qualche secondo per punto, contro le ore della propagazione
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from cnav.uncertainty import (
    OperatingPoint,
    PropagationResult,
    nominal_intensities,
    plot_difference_pdf,
    plot_intensity_pdf,
    plot_points_comparison,
    samples_by_rerun,
    samples_from_result,
)

RESULT_PATH = Path(__file__).resolve().parents[1] / "07_propagazione" / "propagation_result.npz"
OUT_DIR = Path(__file__).resolve().parent

RIVALUTA = False        # True = punti esatti invece dei nodi di griglia
NOMINALE = True         # sovrappone la riga verticale del modello a theta fissato
# Quale theta fissato. "calibrato" = la mediana di theta, cioè il centro
# della nuvola propagata: lo scarto rispetto alla mediana dei campioni è
# allora solo non linearita'. "articolo" = i default delle dataclass, cioè
# il modello prima della calibrazione e prima del fit OEW: utile come
# riferimento storico, ma lo scarto che mostra è in gran parte la
# calibrazione, non l'incertezza
RIFERIMENTO = "calibrato"

# Percentili entro cui calcolare la sagoma dei violini: le code che
# esplodono vicino a un limite di fattibilità schiacciano altrimenti
# tutti i violini in schegge piatte. None per disegnare tutto
CLIP_VIOLINI = (1, 99)

# I punti operativi. Gli stessi nomi di MISSION_LIBRARY
# (cnav/sensitivity_global/outputs.py) così i grafici di questa fase e
# quelli della sensibilità globale si riferiscono agli stessi punti.
# log_x serve dove i sistemi mostrati stanno su ordini di grandezza
# diversi: tipicamente ai raggi corti, dove qualche configurazione
# diverge vicino al proprio limite di fattibilità
PUNTI = [
    (OperatingPoint(100.0, 350.0, "fan", "short_low_speed_fan"), True),
    (OperatingPoint(100.0, 450.0, "fan", "short_high_speed_fan"), True),
    (OperatingPoint(1500.0, 450.0, "fan", "medium_fan"), True),
    (OperatingPoint(6000.0, 450.0, "fan", "long_fan"), True),
    (OperatingPoint(100.0, 200.0, "propeller", "short_low_speed_propeller"), True),
    (OperatingPoint(100.0, 300.0, "propeller", "short_high_speed_propeller"), True),
    (OperatingPoint(1500.0, 250.0, "propeller", "medium_propeller"), True),
    (OperatingPoint(6000.0, 250.0, "propeller", "long_propeller"), True),
]

# I sistemi da seguire nel grafico a violini (uno per sistema)
SISTEMI_VIOLINI = ["Hydrogen combustion (fan)", "e-SAF combustion (fan)", "Battery-electric (fan)", "Hydrogen fuel cell (fan)",
                   "Hydrogen combustion (propeller)", "e-SAF combustion (propeller)", "Battery-electric (propeller)", 
                   "Hydrogen fuel cell (propeller)"]


def _slug(label: str) -> str:
    """Stessa convenzione di plot_probabilistic_technology_map.py"""
    return (label.lower().replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", ""))


def main():
    result = PropagationResult.load(RESULT_PATH)
    print(f"Caricati {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")
    if RIVALUTA:
        print("Modo esatto: i punti vengono rivalutati sul theta salvato "
              "(qualche secondo per punto)")

    campioni, tabelle = [], []

    for punto, log_x in PUNTI:
        print(f"\n{punto.title}")
        if RIVALUTA:
            s = samples_by_rerun(result.theta, punto)
        else:
            s = samples_from_result(result, punto)
        campioni.append(s)

        tab = s.summary()
        tab.insert(0, "punto", punto.name or punto.slug)
        tabelle.append(tab)
        print(tab.drop(columns=["punto"]).round(3).to_string(index=False))

        # --- 1. le PDF sovrapposte ------------------------------------
        overrides = (result.theta.median().to_dict()
                     if RIFERIMENTO == "calibrato" else None)
        nominale = nominal_intensities(punto, overrides=overrides) if NOMINALE else None
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        try:
            plot_intensity_pdf(s, ax=ax, log_x=log_x, nominal=nominale)
        except ValueError as err:
            print(f"  (nessuna figura: {err})")
            plt.close(fig)
            continue
        fig.tight_layout()
        path = OUT_DIR / f"output_pdf_{punto.slug}.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}")
        plt.close(fig)

        # --- 2. la differenza fra i due sistemi più probabili --------
        # i due da confrontare: il più probabile e il suo primo inseguitore.
        # A parità di P_migliore (tipicamente 0, quando un sistema domina il
        # punto) decide la mediana, altrimenti finirebbe in classifica una
        # configurazione mai fattibile, che non è un inseguitore
        candidati = (tab[tab["n_fattibili"] >= 2]
                     .sort_values(["P_migliore", "p50"], ascending=[False, True]))
        if len(candidati) < 2:
            print("  (un solo sistema fattibile: nessun confronto da fare)")
            continue
        primo, secondo = candidati["sistema"].tolist()[:2]
        fig, ax = plt.subplots(figsize=(8.5, 5))
        try:
            plot_difference_pdf(s, primo, secondo, ax=ax)
            fig.tight_layout()
            path = OUT_DIR / f"output_diff_{punto.slug}.png"
            fig.savefig(path, dpi=150)
            print(f"  {path.name}")
        except ValueError as err:
            print(f"  (nessun confronto {primo} / {secondo}: {err})")
        plt.close(fig)

    # --- 3. lo stesso sistema in tutti i punti -------------------------
    for sistema in SISTEMI_VIOLINI:
        propulsore = sistema.rsplit(" (", 1)[1].rstrip(")")
        sottoinsieme = [s for s in campioni if s.point.propulsor == propulsore]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        try:
            plot_points_comparison(sottoinsieme, sistema, ax=ax,
                                   clip_percentiles=CLIP_VIOLINI)
            fig.tight_layout()
            # il propulsore deve entrare nel nome: senza, la figura a elica
            # sovrascrive quella con fan della stessa tecnologia
            path = OUT_DIR / f"output_points_{_slug(sistema)}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\n  {path.name}")
        except ValueError as err:
            print(f"\n  (nessun violino per {sistema}: {err})")
        plt.close(fig)

    # --- 4. i numeri --------------------------------------------------
    csv = pd.concat(tabelle, ignore_index=True)
    path = OUT_DIR / "output_distributions.csv"
    csv.to_csv(path, index=False)
    print(f"  {path.name}")


if __name__ == "__main__":
    main()