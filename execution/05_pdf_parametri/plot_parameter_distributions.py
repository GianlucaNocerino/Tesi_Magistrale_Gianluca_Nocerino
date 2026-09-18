"""
Le distribuzioni dei parametri incerti (l'ingresso, non l'uscita).

Il complemento di execution/08_mappa_probabilistica/plot_output_distributions.py:
lì si guarda la PDF dell'electricity intensity, qui quella dei parametri
da cui nasce.

Uso:
    python execution/05_pdf_parametri/plot_parameter_distributions.py

Non è solo una figura descrittiva: sovrapponendo la PDF dichiarata al
campione effettivamente propagato si verifica che il campionamento abbia
fatto quello che si voleva. Un istogramma che non segue la curva, o una
correlazione che nel campione non c'è, sono errori che altrimenti
restano invisibili fino a quando non producono un risultato strano tre
fasi più avanti.

Le tre categorie in cui si dividono le colonne di theta (dichiarata,
derivata, empirica) e il perchè la curva teorica si disegni solo sulla
prima sono spiegati in cnav.uncertainty.distributions.

Produce:
  1. parameter_distributions.png   un pannello per parametro
  2. parameter_correlations.png    le coppie con dipendenza imposta
  3. parameter_sample_check.csv    dichiarato contro realizzato
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt

from cnav.model.constants import TechAssumptions, WellToTankEfficiencies
from cnav.uncertainty import (
    PropagationResult,
    assemble_theta,
    build_default_correlations,
    build_default_specs,
    load_theta_acc,
    parameter_sample_check,
    plot_parameter_correlations,
    plot_parameter_distributions,
)

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR.parent / "07_propagazione" / "propagation_result.npz"
THETA_ACC_PATH = OUT_DIR.parent / "06_incertezza_calibrazione" / "theta_acc.csv"

# Se il risultato della propagazione c'è si usa la SUA theta: è il
# campione che ha davvero prodotto le mappe, non uno equivalente. Il
# ricampionamento serve solo per guardare le PDF prima di aver lanciato
# la Fase 7, e costa secondi (è solo LHS, nessuna valutazione del modello)
N_SAMPLES = 1500
SEED = 0
KS_SOGLIA = 0.01        # sotto questo p-value il campione non viene dalla PDF dichiarata


def carica_theta():
    if RESULT_PATH.exists():
        return PropagationResult.load(RESULT_PATH).theta, "propagazione"
    specs = build_default_specs()
    theta = assemble_theta(N_SAMPLES, load_theta_acc(THETA_ACC_PATH), specs=specs,
                           correlations=build_default_correlations(specs), seed=SEED)
    return theta, "ricampionamento"


def nominali_di_calibrazione(theta, specs):
    """I nominali delle colonne senza spec: stanno nelle dataclass del
    modello, non nella tabella delle PDF"""
    fuori = [c for c in theta.columns if c not in specs]
    tech, wtt = TechAssumptions(), WellToTankEfficiencies()
    return {c: float(getattr(tech, c, getattr(wtt, c, float("nan")))) for c in fuori}


def main():
    theta, origine = carica_theta()
    specs = build_default_specs()
    corr = build_default_correlations(specs)
    print(f"theta: {theta.shape[0]} campioni x {theta.shape[1]} parametri (da {origine})")

    fig, _ = plot_parameter_distributions(
        theta, specs, corr, nominals=nominali_di_calibrazione(theta, specs),
        title=f"Distribuzioni dei parametri incerti - {len(theta)} campioni "
              f"({origine}).  Tratteggiata: valore nominale")
    path = OUT_DIR / "parameter_distributions.png"
    fig.savefig(path, dpi=150)
    print(f"  {path.name}")
    plt.close(fig)

    fig, _ = plot_parameter_correlations(theta, corr)
    if fig is not None:
        path = OUT_DIR / "parameter_correlations.png"
        fig.savefig(path, dpi=150)
        print(f"  {path.name}")
        plt.close(fig)

    df = parameter_sample_check(theta, specs, corr)
    path = OUT_DIR / "parameter_sample_check.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}")

    fuori = df[df.get("fuori_supporto", False)]
    if not fuori.empty:
        print("\n  ATTENZIONE, campione fuori dal supporto dichiarato:")
        print(fuori[["parametro", "min_dichiarato", "min_campione",
                     "max_dichiarato", "max_campione"]].to_string(index=False))

    sospetti = df[df.get("ks_p", 1.0) < KS_SOGLIA]
    if not sospetti.empty:
        print(f"\n  ATTENZIONE, campione incompatibile con la marginale "
              f"dichiarata (KS p < {KS_SOGLIA}):")
        print(sospetti[["parametro", "min_dichiarato", "min_campione",
                        "max_dichiarato", "max_campione", "ks_p"]].to_string(index=False))
        print("  di norma significa che il campione e' stato prodotto con una "
              "versione precedente delle specs")


if __name__ == "__main__":
    main()
