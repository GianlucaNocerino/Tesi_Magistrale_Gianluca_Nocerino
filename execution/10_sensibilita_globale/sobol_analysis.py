"""
Analisi di Sobol sui fattori selezionati dopo lo screening di Morris.

Presuppone theta_acc.csv e morris_screening.csv (prodotto da
execution/10_sensibilita_globale/morris_screening.py: serve solo per il confronto finale fra i
due ranking, non per il calcolo).

Uso:
    python execution/10_sensibilita_globale/sobol_analysis.py

PRIMA DI LANCIARLO: si sceglie SELECTED_FACTORS qui sotto, a mano,
guardando morris_clouds.png e morris_cloud_cumulativa.png. Non e' un
passaggio da automatizzare, è la decisione dell'analista, e sul costo
pesa direttamente: N*(g+2) valutazioni con g = fattori scelti + 1.

Produce:
  sobol_indices.csv        tabella tidy (output x fattore: S1, ST, IC, n_valid)
  sobol_convergence.csv    gli stessi indici a N crescente
  sobol_bars.png           S_i e S_T per punto operativo
  sobol_aggregate.png      la media sui punti operativi
  sobol_convergence.png    la prova di convergenza

Come si legge il risultato, nell'ordine:

  0. tail_report: se un output ha la coda lunga, la sua varianza è
     decisa da pochi campioni quasi infattibili e gli indici sono
     instabili (si vedono S_T > 1). Si sposta il punto operativo o si
     usa TRANSFORM = "log"
  1. sobol_summary: somma_S1 vicino a 1 = modello additivo, molto sotto
     1 = varianza nelle interazioni. E frazione_valida_min: se è
     bassa gli indici parlano solo delle configurazioni fattibili
  2. screening_check: ST del gruppo "resto". Piccolo = lo screening ha
     scartato roba irrilevante e la selezione manuale regge. Grosso =
     va allargato il sottoinsieme e rilanciato
  3. i singoli indici, e soprattutto ST - S1: è lì che si vede
     quali fattori contano attraverso le interazioni, cioè il
     risultato che Morris poteva solo segnalare con sigma alto
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from cnav.uncertainty import load_theta_acc
from cnav.sensitivity_global import (
    FactorSpace,
    compare_with_morris,
    default_outputs,
    evaluate_outputs,
    nan_report,
    plot_all_sobol_bars,
    plot_sobol_aggregate,
    plot_sobol_convergence,
    screening_check,
    sobol_convergence,
    sobol_design,
    sobol_indices,
    sobol_summary,
    sobol_table,
    tail_report,
)

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parent       # tutti gli output stanno qui
THETA_ACC_PATH = FIGURE_DIR.parent / "06_incertezza_calibrazione" / "theta_acc.csv"
MORRIS_CSV = FIGURE_DIR / "morris_screening.csv"     # solo per il confronto finale

# I FATTORI SCELTI A MANO leggendo le nubi di Morris.
#
# Questa lista è la decisione dell'analista e va motivata in tesi, una
# riga per fattore. Il criterio usato per la proposta di partenza era
# la distanza dall'origine sulla nube più la dominanza in almeno un
# punto operativo (select_factors), ma la lista finale è più corta:
# ogni fattore in più costa N valutazioni, e i fattori sotto il 10%
# del dominante non spostano il risultato.
#
# Tutti quelli non elencati qui restano campionati dalle loro PDF e
# finiscono nel gruppo "resto": la varianza totale resta quella vera e
# ST[resto] dice se qualcosa di importante e' stato scartato.
SELECTED_FACTORS = [
    # struttura: dominano tutti i punti con fan
    "oew_fan_b",
    "oew_fan_r_pivot",
    # sigma alto nelle nubi: è il candidato principale per S_T - S_1
    "eta_motor",
    # quanto pesa l'incertezza residua di calibrazione rispetto a quella
    # di letteratura: è una domanda della tesi, non solo un fattore
    "calibration_block",
    # i dominanti di ciascun sistema. Vanno tenuti anche se in media
    # contano poco: ognuno governa il suo pezzo di mappa, ed è proprio
    # la ripartizione per sistema che interessa
    "e_battery_Wh_per_kg",                  # batteria
    "fuel_cell_specific_power_kW_per_kg",   # fuel cell
    "eta_fuel_cell",                        # fuel cell
    "liquid_hydrogen",                      # H2: include il fattore comune di elettrolisi
    "e_saf",                                # e-SAF: residuo idiosincratico
    "gamma_tank",                           # serbatoi criogenici
]

# Restano fuori: eta_p_fan, eta_p_propeller, oew_prop_b, oew_prop_r_pivot, electricity.
# Se screening_check li promuove, si rilancia con quel fattore aggiunto.
#
# NOTA sul costo: qui i fattori totali sono 15 e il modello costa ~25 ms
# a valutazione, quindi la Sobol completa (N*17) costerebbe solo il 30%
# in più di questa. Lo screening, su questo modello, serve più a
# giustificare la scelta e a tenere leggibili i grafici che a
# risparmiare tempo: vale la pena dirlo in tesi invece di far finta che
# fosse una necessità di calcolo.

N_BASE = 2048               # potenza di 2. 512 per provare, 1024-4096 per la tesi
N_BOOTSTRAP = 200          # 0 per saltare gli intervalli di confidenza
SEED = 1

TOLLERANZA_RESTO = 0.05    # soglia di screening_check: 5% della varianza

# Trasformazione della risposta prima di decomporne la varianza.
#
#   None    varianza di Y: "chi governa gli MJ/pax/nmi"
#   "log"   varianza di log(Y): "chi governa le variazioni RELATIVE"
#
# Qui serve "log". Diversi punti operativi stanno vicino al confine di
# fattibilità del loro sistema, e lì l'intensity non va a NaN di
# colpo: prima esplode (il punto fisso del sizing converge a MTOW
# enormi). Il rapporto massimo/mediana arriva a 30 (vedi tail_report
# qui sotto), la varianza finisce decisa da una manciata di campioni e
# senza log si ottengono S_T maggiori di 1, che non hanno senso.
# La scelta va dichiarata in tesi: cambia la domanda, e su grandezze
# che spaziano ordini di grandezza è anche quella più sensata
TRANSFORM = "log"

OUTPUTS = default_outputs()
INDICES_CSV = FIGURE_DIR / "sobol_indices.csv"
CONVERGENCE_CSV = FIGURE_DIR / "sobol_convergence.csv"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)

    theta_acc = load_theta_acc(THETA_ACC_PATH)
    space = FactorSpace.default(theta_acc)
    design = sobol_design(space, SELECTED_FACTORS, N=N_BASE, seed=SEED)

    print("=" * 78)
    print("DISEGNO")
    print("=" * 78)
    print(design.cost_table().to_string(index=False))
    scartati = [nm for nm in space.names if nm not in SELECTED_FACTORS]
    print(f"\nAnalizzati singolarmente: {len(SELECTED_FACTORS)} fattori")
    for nm in SELECTED_FACTORS:
        print(f"  - {nm}")
    print(f"\nRaggruppati in 'resto' ({len(scartati)}), campionati ma non separati:")
    for nm in scartati:
        print(f"  - {nm}")
    print("\nRestano campionati apposta: fissarli al nominale ridurrebbe la")
    print("varianza totale e gonfierebbe tutti gli altri indici.")

    # -----------------------------------------------------------------
    # Valutazione
    # -----------------------------------------------------------------
    theta = space.theta_from_unit(design.U)

    print("\n" + "=" * 78)
    print("VALUTAZIONE DEL MODELLO")
    print("=" * 78)
    print(f"{design.n_evaluations} righe x {len(OUTPUTS)} punti operativi")

    t0 = time.time()
    Y = evaluate_outputs(theta, OUTPUTS, verbose=True, progress_every=500)
    dt = time.time() - t0
    print(f"Tempo: {dt:.0f}s ({dt / len(theta):.3f}s per valutazione)")

    print("\nOutput non definiti (da guardare prima degli indici):")
    print(nan_report(Y).to_string())

    print("\nLunghezza delle code (massimo / mediana sul campione A+B):")
    print(tail_report(design, Y).round(2).to_string(index=False))
    print("\n  'coda lunga' = la varianza è decisa da pochi campioni vicini al")
    print("  confine di fattibilità. O si sposta il punto operativo dentro il")
    print(f"  dominio, o si decompone log(Y): qui TRANSFORM = {TRANSFORM!r}")

    # -----------------------------------------------------------------
    # Indici
    # -----------------------------------------------------------------
    df = sobol_indices(design, Y, n_boot=N_BOOTSTRAP, seed=SEED, transform=TRANSFORM)
    df.to_csv(INDICES_CSV, index=False)
    print(f"\nTabella completa salvata in {INDICES_CSV}")

    print("\n" + "=" * 78)
    print("DIAGNOSTICHE PER OUTPUT")
    print("=" * 78)
    print(sobol_summary(df).round(3).to_string(index=False))
    print("\n  somma_S1 << 1 = varianza nelle interazioni (atteso: il sizing è")
    print("  un punto fisso, ogni parametro rientra nel MTOW e quindi in tutto)")
    print("  S1_min molto negativo = rumore di campionamento, alzare N")

    print("\n" + "=" * 78)
    print(f"VERIFICA DELLO SCREENING (tolleranza {TOLLERANZA_RESTO:.0%})")
    print("=" * 78)
    check = screening_check(df, tolleranza=TOLLERANZA_RESTO)
    print(check.round(3).to_string(index=False))
    da_rivedere = check[check["esito"] == "da rivedere"]
    if len(da_rivedere):
        print(f"\n  {len(da_rivedere)} punti operativi sopra tolleranza: in quei punti")
        print("  i fattori scartati contano. Guardare la nube di Morris di QUEL")
        print("  punto, aggiungere il fattore appena sotto soglia e rilanciare")
    else:
        print("\n  Tutti sotto tolleranza: la selezione manuale regge")

    print("\n" + "=" * 78)
    print("INDICI TOTALI S_T (fattori x punti operativi)")
    print("=" * 78)
    print(sobol_table(df, "ST").round(3).to_string())

    print("\n" + "=" * 78)
    print("QUOTA DI INTERAZIONE S_T - S_1")
    print("=" * 78)
    print(sobol_table(df, "interazione").round(3).to_string())
    print("\n  Valori alti = il fattore conta soprattutto insieme ad altri.")
    print("  Da confrontare con sigma delle nubi di Morris: è la stessa")
    print("  informazione, qui pero' quantificata")

    # -----------------------------------------------------------------
    # Convergenza
    # -----------------------------------------------------------------
    conv = sobol_convergence(design, Y, seed=SEED, transform=TRANSFORM)
    conv.to_csv(CONVERGENCE_CSV, index=False)

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------
    fig = plot_all_sobol_bars(df, ncols=3)
    bars_path = FIGURE_DIR / "sobol_bars.png"
    fig.savefig(bars_path, dpi=140)

    ax = plot_sobol_aggregate(df)
    ax.legend(fontsize=15, loc="lower right")
    agg_path = FIGURE_DIR / "sobol_aggregate.png"
    ax.figure.savefig(agg_path, dpi=140)

    ax = plot_sobol_convergence(conv, index="ST")
    ax.legend(fontsize=7.5, ncol=2, loc="best")
    conv_path = FIGURE_DIR / "sobol_convergence.png"
    ax.figure.savefig(conv_path, dpi=140)

    # -----------------------------------------------------------------
    # Morris vs Sobol
    # -----------------------------------------------------------------
    morris_path = Path(MORRIS_CSV)
    if morris_path.exists():
        print("\n" + "=" * 78)
        print("MORRIS vs SOBOL (ranking medio sugli output)")
        print("=" * 78)
        print(compare_with_morris(df, pd.read_csv(morris_path)).round(3).to_string())
        print("\n  delta_rango vicino a zero = lo screening ordinava bene, e la")
        print("  Sobol ha confermato con r*(k+1) valutazioni in meno. Righe con")
        print("  rango_sobol vuoto = fattori scartati, mai quantificati")
    else:
        print(f"\n({MORRIS_CSV} non trovato: salto il confronto con Morris)")

    print(f"\nFigure salvate in {FIGURE_DIR}:")
    for p in (bars_path, agg_path, conv_path):
        print(f"  {p.name}")

    plt.close("all")


if __name__ == "__main__":
    main()
