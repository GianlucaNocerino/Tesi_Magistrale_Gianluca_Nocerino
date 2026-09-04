"""
Dal campione theta alla mappa probabilistica.

Presuppone che esista theta_acc.csv 
(prodotto da examples/calibration_uncertainty_execution.py).

Uso (fai PRIMA un giro con N_SAMPLES piccolo!):
    python examples/uncertainty_propagation_execution.py

Il risultato grezzo viene salvato in propagation_result.npz: tutte le
analisi (mappe, soglie, confini, bande di confidenza) si
rifanno da quel file senza rilanciare il modello. Vedi
examples/plot_probabilistic_technology_map.py.

"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cnav.uncertainty import (
    assemble_theta,
    build_default_correlations,
    build_default_specs,
    load_theta_acc,
    induced_spec,
    induced_spec_many,
    specs_table,
    FlightGrid,
    run_propagation,
    probability_map,
    aggregate_to_technologies,
    robust_area_fraction,
    boundary_statistics,
    adjacent_label_pairs,
    feasibility_probability,
)

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
THETA_ACC_PATH = "theta_acc.csv"       # output della Fase 6
N_SAMPLES = 1500                        # <-- 200 per provare, 1000+ per la tesi
SEED = 0
N_WORKERS = 1                          # >1 richiede il guard __main__ (vedi sotto)
CHECKPOINT_DIR = "propagation_checkpoints"
CHUNK_SIZE = 25
MIN_CELLS = 3                          # celle di contatto minime per considerare
                                       # due tecnologie adiacenti sulla mappa
OUTPUT_PATH = "propagation_result.npz"

GRID = FlightGrid.default(n_ranges=45, n_speeds=40)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    # -----------------------------------------------------------------
    # Tabella delle PDF
    # -----------------------------------------------------------------
    specs = build_default_specs()
    corr = build_default_correlations(specs)
    print("=" * 78)
    print("TABELLA DELLE DISTRIBUZIONI")
    print("=" * 78)
    print(specs_table(specs, corr)[["parametro", "nominale", "min", "max", "moda", "PDF"]]
          .round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("CORRELAZIONI FISICHE IMPOSTE")
    print("=" * 78)
    for dp in corr.derived:
        ind = induced_spec(dp, specs[dp.driver])
        print(f"  [derivato]          {dp.name} = {dp.relation_label}")
        print(f"      {dp.rationale}")
        print(f"      marginale indotta: min={ind['min']:.4f}, moda={ind.get('moda', float('nan')):.4f}, "
              f"max={ind['max']:.4f}  (non quella ipotizzata: è questa che va in tabella)")
    for dp in corr.derived_many:
        ind = induced_spec_many(dp, specs)
        print(f"  [derivato da più]   {dp.name} = {dp.relation_label}")
        print(f"      {dp.rationale}")
        print(f"      marginale indotta: min={ind['min']:.4f}, media={ind['media']:.4f}, "
              f"max={ind['max']:.4f}, sd={ind['sd']:.4f}  (stimata per campionamento)")
    for cp in corr.copulas:
        print(f"  [copula gaussiana]  {cp.a} <-> {cp.b}  (rho = {cp.rho:.3f})")
        print(f"      {cp.rationale}")

    # -----------------------------------------------------------------
    # Assemblaggio di theta: letteratura + righe di Theta_acc
    # -----------------------------------------------------------------
    theta_acc = load_theta_acc(THETA_ACC_PATH)
    print(f"\nTheta_acc: {len(theta_acc)} righe accettate, "
          f"{len(theta_acc.columns)} parametri di calibrazione")

    theta = assemble_theta(N_SAMPLES, theta_acc, specs=specs,
                           correlations=corr, seed=SEED)
    print(f"Matrice theta: {theta.shape[0]} campioni x {theta.shape[1]} parametri")
    print(f"  letteratura:  {list(specs.keys())}")
    print(f"  calibrazione: {list(theta_acc.columns)}")

    # correlazioni effettivamente realizzate nel campione: controllo che
    # quello che si voleva imporre sia stato imposto davvero
    print("\nCorrelazioni realizzate nel campione:")
    for dp in corr.derived:
        r = theta[dp.driver].corr(theta[dp.name])
        print(f"  {dp.driver} / {dp.name}: r = {r:+.3f}  (atteso esattamente -1 o +1)")
    for dp in corr.derived_many:
        for drv in dp.drivers:
            r = theta[drv].corr(theta[dp.name])
            print(f"  {drv} / {dp.name}: r = {r:+.3f}  (relazione deterministica, "
                  f"non lineare: |r| < 1 è atteso)")
        if len(dp.drivers) == 2:
            r_dd = theta[dp.drivers[0]].corr(theta[dp.drivers[1]])
            print(f"  {dp.drivers[0]} / {dp.drivers[1]}: r = {r_dd:+.3f}  "
                  f"(atteso ~0: è il presupposto del cambio di coordinate)")
    for cp in corr.copulas:
        r = theta[cp.a].corr(theta[cp.b])
        print(f"  {cp.a} / {cp.b}: r = {r:+.3f}  (target {cp.rho:+.3f})")

    # -----------------------------------------------------------------
    # Propagazione
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PROPAGAZIONE")
    print("=" * 78)
    print(f"Griglia: {len(GRID.ranges_nmi)} range x {len(GRID.speeds_kt)} velocità "
          f"= {GRID.n_model_evaluations()} valutazioni per campione")

    t0 = time.time()
    result = run_propagation(theta, GRID, checkpoint_dir=CHECKPOINT_DIR,
                             chunk_size=CHUNK_SIZE, n_workers=N_WORKERS,
                             resume=True, verbose=True)
    print(f"Tempo totale: {time.time() - t0:.0f}s")

    result.save(OUTPUT_PATH)
    print(f"Risultato grezzo salvato in {OUTPUT_PATH}")

    # -----------------------------------------------------------------
    # Mappa probabilistica
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MAPPA PROBABILISTICA")
    print("=" * 78)

    pmap = probability_map(result)
    agg = aggregate_to_technologies(pmap)

    print("\nFrazione di griglia con una tecnologia dominante, al variare della soglia:")
    print("  (8 etichette, tecnologia x propulsore)")
    print(robust_area_fraction(pmap).round(3).to_string(index=False))
    print("  (4 tecnologie, fan ed elica aggregati)")
    print(robust_area_fraction(agg).round(3).to_string(index=False))

    print("\nFrazione di griglia in cui ciascuna tecnologia è la più probabile:")
    dominant = agg.dominant_index
    for j, label in enumerate(agg.labels):
        share = float((dominant == j).mean())
        p_max_where = agg.probabilities[j][dominant == j]
        p_mean = float(p_max_where.mean()) if p_max_where.size else float("nan")
        print(f"  {label:22s} {100 * share:5.1f}% della griglia   "
              f"P media dove domina: {p_mean:.2f}")

    print("\nFattibilità della batteria (probabilità massima sulla griglia):")
    for label in ("Battery-electric (fan)", "Battery-electric (propeller)"):
        print(f"  {label:32s} max P(fattibile) = {feasibility_probability(result, label).max():.2f}")

    # -----------------------------------------------------------------
    # Dispersione dei confini
    # -----------------------------------------------------------------
    # Le coppie non sono scelte a mano. boundary_statistics cerca
    # l'incrocio fra due curve di intensity senza verificare che una
    # delle due stia effettivamente vincendo: su una coppia che sulla
    # mappa non si tocca mai produce un confine calcolabile e privo di
    # significato, perche' sepolto sotto una terza tecnologia.
    print("\n" + "=" * 78)
    print("DISPERSIONE DEI CONFINI")
    print("=" * 78)

    pairs = adjacent_label_pairs(pmap, min_cells=MIN_CELLS)
    if not pairs:
        print("Nessuna coppia adiacente: un'unica tecnologia domina tutta la griglia.")
    for label_a, label_b, n_cells in pairs:
        print(f"\n{label_a}  /  {label_b}")
        print(f"  (domini a contatto su {n_cells} celle della griglia)")
        bstats = boundary_statistics(result, label_a, label_b)
        print(bstats.round(1).to_string(index=False))

    print("\n  Guarda sempre 'frazione_con_confine' prima delle altre colonne: dove è")
    print("  bassa, media e percentili sono calcolati su una minoranza di campioni")

    # etichette che non vincono mai: se una tecnologia attesa finisce qui,
    # e' un risultato da spiegare, non un dettaglio
    mai = [l for l in pmap.labels if pmap.probability_of(l).max() < 0.01]
    if mai:
        print("\n  Mai la migliore in nessun campione e in nessun punto:")
        for l in mai:
            print(f"    - {l}")


if __name__ == "__main__":
    # il guard serve davvero: con N_WORKERS > 1 i processi figli
    # reimportano questo file, e senza guard rieseguirebbero tutto
    main()