"""
"Uncertainty from calibration".

Per i parametri che non hanno una PDF direttamente giustificabile dalla
letteratura, ma che sono stati introdotti o modificati durante la
calibrazione, l'incertezza va dedotta dal processo di
calibrazione stesso, campionando la regione accettabile

    Theta_acc = {theta : J(theta) <= J_thr}

invece di assegnare a mano una PDF (Gaussiana o altro) attorno a theta*.
Il metodo pratico indicato è:

    1. bounds fisicamente plausibili (già quelli usati per calibrare,
       vedi deterministic_calibration.default_bounds/manual_bounds);
    2. campione via Latin Hypercube Sampling;
    3. valutazione di J(theta) su tutto il campione;
    4. selezione dei punti con J <= J_thr;
    5. analisi della distribuzione dei punti accettati
    6. i punti accettati (l'intera matrice, non le sue statistiche
       marginali) sono i campioni da usare direttamente nella successiva
       uncertainty propagation: per i parametri di
       calibrazione, invece di campionare da una PDF, si pesca a caso una
       RIGA di Theta_acc, preservando la struttura di correlazione.

NOTA SUL COSTO COMPUTAZIONALE (importante prima di lanciare N=10**4):
ogni valutazione di J costa quanto nella calibrazione deterministica
(indicativamente 0.3-0.6s per combined_cost, vedi il docstring di
run_deterministic_calibration). Con N=10_000 campioni, un run SEQUENZIALE
può quindi richiedere 1-2 ore. Per questo evaluate_J_over_sample scrive
un checkpoint su disco ogni checkpoint_every campioni (default 500): se
l'esecuzione si interrompe (kernel killato, laptop chiuso, ...), si
riparte da dove si era arrivati con resume_from_checkpoint=True invece di
perdere tutto. Consiglio pratico dato dalla stessa Sezione 7: fare un
primo giro con N piccolo (500-1000) per verificare che il flusso
funzioni e stimare il tempo per campione, poi alzare a 10**4 per il
risultato "vero" da riportare in tesi, girato offline/in background (non
in una sessione interattiva), esattamente come per la calibrazione
deterministica.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .deterministic_calibration import latin_hypercube_starts

__all__ = [
    "j_thr_from_margin",
    "evaluate_J_over_sample",
    "select_acceptable_region",
    "CalibrationUncertaintyResult",
    "run_calibration_uncertainty",
    "accepted_correlation_matrix",
    "sample_calibrated_theta",
]


# ---------------------------------------------------------------------
# Soglia di accettabilità J_thr (par. 7: "deve essere giustificato
# sulla base di una tolleranza di riproduzione ritenuta accettabile")
# ---------------------------------------------------------------------

def j_thr_from_margin(j_star: float, margin: float = 0.10) -> float:
    """J_thr = j_star * (1 + margin), il modo più semplice, e quello
    esplicitamente suggerito come prassi, per fissare la soglia: un
    margine (tipicamente 5-20%) sopra il minimo globale trovato in fase
    di calibrazione deterministica.

    j_star deve essere il costo minimo trovato (best_run.cost da
    run_deterministic_calibration), non un valore inventato: la soglia
    è ancorata al risultato della propria calibrazione, non a un numero
    di riferimento esterno.
    """
    if j_star <= 0:
        raise ValueError(f"j_star deve essere positivo, ricevuto {j_star}")
    if margin <= 0:
        raise ValueError(f"margin deve essere positivo, ricevuto {margin}")
    return j_star * (1.0 + margin)


# ---------------------------------------------------------------------
# Campionamento LHS + valutazione di J, con checkpoint su disco
# ---------------------------------------------------------------------

def evaluate_J_over_sample(J: Callable, param_names, theta_samples: np.ndarray,
                            checkpoint_path: Optional[str] = None,
                            checkpoint_every: int = 500,
                            resume_from_checkpoint: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
    """Valuta J(theta) per ogni riga di theta_samples (shape (N, n_theta),
    stesso ordine di param_names), con checkpoint incrementale.

    checkpoint_path: se dato, il DataFrame dei risultati (colonne
    sample_idx, cost, theta_<nome>...) viene scritto su questo path CSV
    ogni checkpoint_every campioni valutati (append, non riscrittura
    completa - efficiente anche per N grandi).

    resume_from_checkpoint: se True e checkpoint_path esiste già, legge
    i campioni già valutati (per sample_idx) e valuta solo quelli
    mancanti, poi appende. Usalo per riprendere un run interrotto SENZA
    ripetere le valutazioni già fatte - fondamentale visto il costo
    (vedi docstring del modulo) di un run con N=10**4.

    Ritorna il DataFrame completo (tutti gli N campioni, non solo quelli
    accettati - il filtraggio è un passo separato, select_acceptable_region,
    così puoi rivedere J_thr senza dover rivalutare il modello).
    """
    n_samples = theta_samples.shape[0]
    already_done = set()

    if resume_from_checkpoint and checkpoint_path is not None and Path(checkpoint_path).exists():
        existing = pd.read_csv(checkpoint_path)
        already_done = set(existing["sample_idx"].astype(int))
        if verbose:
            print(f"  Checkpoint trovato: {len(already_done)}/{n_samples} campioni già valutati, riprendo da lì")
    else:
        existing = pd.DataFrame()

    rows_buffer = []
    header_written = resume_from_checkpoint and checkpoint_path is not None and Path(checkpoint_path).exists()
    t0 = time.time()
    n_done_this_session = 0
    to_evaluate = [i for i in range(n_samples) if i not in already_done]

    for i in to_evaluate:
        theta_i = theta_samples[i]
        cost = float(J(theta_i))
        row = {"sample_idx": i, "cost": cost}
        row.update({f"theta_{name}": v for name, v in zip(param_names, theta_i)})
        rows_buffer.append(row)
        n_done_this_session += 1

        if checkpoint_path is not None and len(rows_buffer) >= checkpoint_every:
            _flush_checkpoint(rows_buffer, checkpoint_path, header_written)
            header_written = True
            rows_buffer = []

        if verbose and n_done_this_session % checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = elapsed / n_done_this_session
            remaining = len(to_evaluate) - n_done_this_session
            eta_min = rate * remaining / 60.0
            print(f"  [{len(already_done) + n_done_this_session}/{n_samples}] cost={cost:.4g}  "
                  f"({elapsed:.0f}s trascorsi, ~{rate:.2f}s/campione, ETA ~{eta_min:.0f} min)")

    if checkpoint_path is not None and rows_buffer:
        _flush_checkpoint(rows_buffer, checkpoint_path, header_written)
        rows_buffer = []

    if checkpoint_path is not None:
        df_all = pd.read_csv(checkpoint_path).sort_values("sample_idx").reset_index(drop=True)
    else:
        df_all = pd.concat([existing, pd.DataFrame(rows_buffer)], ignore_index=True) \
            .sort_values("sample_idx").reset_index(drop=True)

    if verbose:
        print(f"  Valutazione completata: {len(df_all)}/{n_samples} campioni totali "
              f"({time.time() - t0:.0f}s in questa sessione)")
    return df_all


def _flush_checkpoint(rows_buffer, checkpoint_path, header_written):
    df_chunk = pd.DataFrame(rows_buffer)
    df_chunk.to_csv(checkpoint_path, mode="a" if header_written else "w",
                     header=not header_written, index=False)


# ---------------------------------------------------------------------
# Selezione della regione accettabile
# ---------------------------------------------------------------------

def select_acceptable_region(df_all: pd.DataFrame, j_thr: float) -> pd.DataFrame:
    """Theta_acc = {theta : J(theta) <= J_thr}

    df_all: output di evaluate_J_over_sample (colonne cost, theta_<nome>...).
    Ritorna il sotto-DataFrame dei campioni accettati (stesse colonne),
    ordinato per costo crescente. Non modifica df_all.
    """
    accepted = df_all[df_all["cost"] <= j_thr].sort_values("cost").reset_index(drop=True)
    return accepted


# ---------------------------------------------------------------------
# Orchestratore + diagnostica correlazioni + campionamento a righe
# ---------------------------------------------------------------------

@dataclass
class CalibrationUncertaintyResult:
    """Esito completo del passo di uncertainty-from-calibration."""
    df_all: pd.DataFrame          # tutti gli N campioni valutati
    df_accepted: pd.DataFrame     # sotto-insieme con cost <= j_thr (Theta_acc)
    param_names: list
    j_thr: float
    n_samples: int
    n_accepted: int

    @property
    def acceptance_fraction(self) -> float:
        return self.n_accepted / self.n_samples if self.n_samples else 0.0


def run_calibration_uncertainty(J: Callable, param_names, bounds: dict, j_thr: float,
                                 n_samples: int = 10_000, seed: int = 0,
                                 checkpoint_path: Optional[str] = None,
                                 checkpoint_every: int = 500,
                                 resume_from_checkpoint: bool = False,
                                 verbose: bool = True) -> CalibrationUncertaintyResult:
    """Esegue per intero i passi:

        a. campiona n_samples punti via LHS entro bounds;
        b. valuta J su tutti (con checkpoint, vedi evaluate_J_over_sample);
        c. seleziona Theta_acc = {theta : J(theta) <= j_thr};
        d. impacchetta il risultato (df_accepted è già la
             distribuzione congiunta calibrata da usare in Fase 7 -
             vedi sample_calibrated_theta - senza costruire PDF
             marginali indipendenti, par. 7.1).

    bounds: stesso dict {param_name: (lo, hi)} già usato per la
    calibrazione deterministica (default_bounds/manual_bounds), qui
    non viene ristretto o allargato rispetto a quello, per coerenza con
    "bounds fisicamente plausibili". Se vuoi bounds
    diversi per l'LHS rispetto a quelli usati per trovare theta*, è una
    scelta esplicita da giustificare in tesi, non il default.

    j_thr: soglia di accettabilità, tipicamente da j_thr_from_margin(j_star,
    margin=...) con j_star = best_run.cost della calibrazione deterministica.

    Se acceptance_fraction risulta molto piccola (es. <1%) o molto
    grande (es. >50%), è un segnale da commentare in tesi: una frazione
    troppo piccola può indicare N insufficiente o J_thr troppo stretto
    (Theta_acc mal campionato, poco rappresentativo); una frazione
    troppo grande può indicare J_thr troppo largo (la "regione
    accettabile" diventa poco informativa, quasi tutto il box di bounds
    è accettato).
    """
    bounds = {p: bounds[p] for p in param_names}  # forza l'ordine di param_names
    _, theta_samples = latin_hypercube_starts(bounds, n_samples, seed=seed)

    if verbose:
        print(f"Campionamento LHS: {n_samples} punti su {len(param_names)} parametri")
        print(f"J_thr = {j_thr:.6g}")

    df_all = evaluate_J_over_sample(
        J, param_names, theta_samples,
        checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
        resume_from_checkpoint=resume_from_checkpoint, verbose=verbose,
    )
    df_accepted = select_acceptable_region(df_all, j_thr)

    result = CalibrationUncertaintyResult(
        df_all=df_all, df_accepted=df_accepted, param_names=list(param_names),
        j_thr=j_thr, n_samples=len(df_all), n_accepted=len(df_accepted),
    )

    if verbose:
        print(f"\nTheta_acc: {result.n_accepted}/{result.n_samples} campioni accettati "
              f"({100 * result.acceptance_fraction:.1f}%)")
        if result.acceptance_fraction < 0.01:
            print("  ATTENZIONE: frazione accettata molto piccola - considera di aumentare "
                  "N o rivedere J_thr")
        elif result.acceptance_fraction > 0.5:
            print("  ATTENZIONE: frazione accettata molto grande - J_thr potrebbe essere "
                  "troppo largo per essere informativo")

    return result


def accepted_correlation_matrix(result: CalibrationUncertaintyResult) -> pd.DataFrame:
    """Matrice di correlazione (Pearson) tra i parametri in Theta_acc
    (le correlazioni tra parametri calibrati sono parte del
    risultato, non un dettaglio da nascondere). Da riportare in
    tesi accanto alla distribuzione dei punti accettati"""
    theta_cols = [f"theta_{p}" for p in result.param_names]
    corr = result.df_accepted[theta_cols].corr()
    corr.columns = result.param_names
    corr.index = result.param_names
    return corr


def sample_calibrated_theta(result: CalibrationUncertaintyResult, n: int,
                             seed: Optional[int] = None) -> pd.DataFrame:
    """Campiona n RIGHE (con reinserimento) da Theta_acc, non n valori
    indipendenti per ciascun parametro. Pescando righe intere invece 
    di ricostruire PDF marginali indipendenti, la struttura di 
    correlazione/compensazione tra parametri osservata nella calibrazione 
    (equifinalità) viene preservata nella propagazione dell'incertezza.

    Da usare nel Monte Carlo di propagazione: per ogni campione k della
    propagazione, i parametri "di calibrazione" (quelli in
    result.param_names) si ottengono pescando una riga qui, mentre i
    parametri con una PDF di letteratura (par. 6) si campionano
    separatamente dalla loro distribuzione dichiarata.

    Ritorna un DataFrame di n righe con una colonna per parametro
    (nomi originali, senza il prefisso "theta_").
    """
    if result.n_accepted == 0:
        raise ValueError(
            "Theta_acc è vuoto (nessun campione accettato): non è possibile campionare. "
            "Rivedi J_thr o aumenta n_samples in run_calibration_uncertainty."
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, result.n_accepted, size=n)
    theta_cols = [f"theta_{p}" for p in result.param_names]
    sampled = result.df_accepted.iloc[idx][theta_cols].reset_index(drop=True)
    sampled.columns = result.param_names
    return sampled