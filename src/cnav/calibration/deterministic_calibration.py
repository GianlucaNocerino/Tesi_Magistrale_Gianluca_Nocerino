"""
Calibrazione deterministica iniziale:

    theta* = argmin_theta J(theta)

Questo modulo è un motore di ottimizzazione GENERICO rispetto a J: non sa
nulla di fan_cost/propeller_cost/combined_cost - quelle vivono in
cost_functions.py. Chi chiama sceglie quale funzione di costo calibrare
(fan_cost, propeller_cost, cost_functions.combined_cost, o qualunque altra
funzione con la stessa forma cost_fn(tech, wtt, **kwargs) -> (totale,
dettaglio)), la trasforma in J con make_objective, e passa J a
run_deterministic_calibration. Vedi execution/04_calibrazione/deterministic_calibration_execution.py
per un esempio completo (calibrazione congiunta fan+propeller con
combined_cost).

1. default_bounds / manual_bounds - bounds fisicamente plausibili di
   partenza per l'ottimizzatore, sono solo una scatola entro cui 
   cercare theta*. default_bounds applica una percentuale 
   (uguale per tutti o parametro per parametro) attorno al valore
   nominale; manual_bounds lascia scegliere, parametro per parametro,
   percentuale o bound assoluto
2. theta_to_tech_wtt / make_objective - "adattatore" tra un vettore piatto
   theta (quello che vuole scipy.optimize) e TechAssumptions o
   WellToTankEfficiencies (quello che vuole una cost_fn del tipo
   cost_fn(tech, wtt, **kwargs) -> (totale, dettaglio))
3. run_deterministic_calibration - il solutore vero e proprio: prende una
   J già pronta (costruita con make_objective o a mano) e ci fa girare
   sopra più ottimizzazioni Nelder-Mead da punti iniziali diversi (Latin
   Hypercube) e/o differential_evolution ripetuta con seed diversi, come
   raccomandato ("non utilizzare un solo punto iniziale")
4. summarize_multiple_minima - diagnostica preliminare per l'identificabilità, 
   ossia segnala se più run convergono a costi simili ma per theta* diversi (equifinality).

I parametri da calibrare (param_names) e i loro bounds sono sempre scelti
esplicitamente da chi chiama - dal notebook/script, non da questo modulo:
qui non c'è nessuna selezione automatica a partire dalla sensitivity
analysis (par. 4.1). Se in futuro serve reintrodurla, il ranking di
|elasticità| è comunque disponibile direttamente dal DataFrame restituito
da cnav.sensitivity.local_sensitivity.run_local_sensitivity_report.

NOTA: Poichè la funzione da ottimizzare non è garantita liscia/differenziabile ovunque, 
si è scelto di usare metodi derivative-free (Nelder-Mead, differential
evolution)
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from ..model.constants import TechAssumptions, WellToTankEfficiencies

# parametri "di frazione/efficienza pura", fisicamente in [0, 1]: i bounds
# di default li tengono dentro (0.02, 0.98) invece che lasciarli sconfinare
# con il +-rel_width simmetrico usato per tutti gli altri parametri
FRACTION_PARAMETERS = {
    "eta_motor", "eta_battery", "eta_fuel_cell", "eta_p_propeller", "eta_p_fan",
    "gamma_tank", "battery_oew_fraction", "electricity", "liquid_hydrogen", "e_saf",
}


# ---------------------------------------------------------------------
# Bounds di default per l'ottimizzatore
# ---------------------------------------------------------------------

def default_bounds(param_names, rel_width=0.4,
                    tech: Optional[TechAssumptions] = None,
                    wtt: Optional[WellToTankEfficiencies] = None) -> dict:
    """Bounds simmetrici attorno al valore nominale, [n - w*|n|, n + w*|n|],
    con clipping a (0.02, 0.98) per i parametri di FRACTION_PARAMETERS.

    rel_width può essere:
      - un float (default 0.4, cioè +-40%): stessa percentuale per tutti
        i param_names;
      - un dict {param_name: percentuale}: percentuale diversa per
        ciascun parametro (es. {"eta_motor": 0.1, "e_battery_Wh_per_kg":
        0.5} per un +-10% su eta_motor e +-50% su e_battery_Wh_per_kg).
        Deve avere una voce per ogni nome in param_names (nessun default
        silenzioso "a metà": se un parametro manca, viene sollevato un
        errore così te ne accorgi subito invece di ritrovarti un bound
        non voluto).

    Per bounds assoluti (non percentuali, es. e_battery_Wh_per_kg in
    [250, 600]) non serve questa funzione: costruisci direttamente un
    dict {param_name: (lo, hi)} e passalo a run_deterministic_calibration
    (o usa manual_bounds() sotto per mescolare percentuali e assoluti
    parametro per parametro nella stessa chiamata).

    ATTENZIONE: questi bounds servono solo a mantenere l'ottimizzatore in
    una regione plausibile durante la ricerca di theta* (par. 5.1). Non
    sono le PDF documentate richieste al par. 6.1 (valore nominale,
    intervallo, fonte, tipo di distribuzione, motivazione): quelle vanno
    definite/riviste a mano con riferimenti fisici o di letteratura, non
    riciclate automaticamente da qui.
    """
    tech = tech or TechAssumptions()
    wtt = wtt or WellToTankEfficiencies()
    nominal = {**dataclasses.asdict(tech), **dataclasses.asdict(wtt)}

    if isinstance(rel_width, dict):
        missing = [p for p in param_names if p not in rel_width]
        if missing:
            raise ValueError(
                f"rel_width è un dict ma manca una percentuale per: {missing}. "
                "Specifica una percentuale per ciascun parametro in param_names.")
        width_of = rel_width
    else:
        width_of = {p: rel_width for p in param_names}

    bounds = {}
    for p in param_names:
        if p not in nominal:
            raise ValueError(f"Parametro sconosciuto: {p!r}")
        n = nominal[p]
        w = width_of[p]
        width = w * abs(n) if n != 0 else w
        lo, hi = n - width, n + width
        if p in FRACTION_PARAMETERS:
            lo, hi = max(lo, 0.02), min(hi, 0.98)
        bounds[p] = (lo, hi)
    return bounds


def manual_bounds(spec: dict, tech: Optional[TechAssumptions] = None,
                   wtt: Optional[WellToTankEfficiencies] = None) -> dict:
    """Bounds scelti a mano, parametro per parametro, mescolando liberamente
    due forme nello stesso dict spec:

      - una tupla (lo, hi): bound ASSOLUTO, usato così com'è;
      - un float w: bound PERCENTUALE, [n - w*|n|, n + w*|n|] attorno al
        valore nominale del parametro (stessa convenzione di default_bounds,
        ma richiesta esplicitamente parametro per parametro).

    Esempio:
        bounds = manual_bounds({
            "eta_motor": 0.1,                    # +-10% attorno al nominale
            "e_battery_Wh_per_kg": (250.0, 600.0),  # bound assoluto
            "ld_baseline_fan": 0.25,             # +-25% attorno al nominale
        })

    Nessun parametro "escluso" implicitamente: i parametri calibrati sono
    esattamente e soltanto le chiavi di spec, nell'ordine in cui le
    inserisci (Python 3.7+ preserva l'ordine di un dict).
    """
    tech = tech or TechAssumptions()
    wtt = wtt or WellToTankEfficiencies()
    nominal = {**dataclasses.asdict(tech), **dataclasses.asdict(wtt)}

    bounds = {}
    for p, value in spec.items():
        if p not in nominal:
            raise ValueError(f"Parametro sconosciuto: {p!r}")
        if isinstance(value, tuple):
            lo, hi = value
            if lo >= hi:
                raise ValueError(f"Bound non valido per {p!r}: lo={lo} >= hi={hi}")
        else:
            n = nominal[p]
            width = float(value) * abs(n) if n != 0 else float(value)
            lo, hi = n - width, n + width
            if p in FRACTION_PARAMETERS:
                lo, hi = max(lo, 0.02), min(hi, 0.98)
        bounds[p] = (lo, hi)
    return bounds


# ---------------------------------------------------------------------
# Adattatore theta (vettore piatto) <-> (TechAssumptions, WellToTankEfficiencies)
# ---------------------------------------------------------------------

def theta_to_tech_wtt(theta, param_names,
                       base_tech: Optional[TechAssumptions] = None,
                       base_wtt: Optional[WellToTankEfficiencies] = None):
    """Ricostruisce (tech, wtt) applicando theta (nello stesso ordine di
    param_names) sopra i valori di base (default: nominali)."""
    base_tech = base_tech or TechAssumptions()
    base_wtt = base_wtt or WellToTankEfficiencies()
    tech_kwargs, wtt_kwargs = {}, {}
    for name, value in zip(param_names, theta):
        if hasattr(base_tech, name):
            tech_kwargs[name] = float(value)
        elif hasattr(base_wtt, name):
            wtt_kwargs[name] = float(value)
        else:
            raise ValueError(f"Parametro sconosciuto o non mappato: {name!r}")
    return base_tech.with_changes(**tech_kwargs), base_wtt.with_changes(**wtt_kwargs)


def make_objective(cost_fn: Callable, param_names,
                    base_tech: Optional[TechAssumptions] = None,
                    base_wtt: Optional[WellToTankEfficiencies] = None,
                    cost_kwargs: Optional[dict] = None,
                    penalty_on_error: float = 1e6):
    """Costruisce J: R^n_theta -> R da passare a scipy.optimize, a partire
    da una QUALSIASI funzione di costo cost_fn(tech, wtt, **cost_kwargs) ->
    (totale, dettaglio) - tipicamente cnav.calibration.cost_functions.fan_cost,
    propeller_cost, o combined_cost, ma può essere una qualunque funzione
    con quella forma (anche definita altrove, non necessariamente in
    cost_functions.py).

    Questo modulo non ha nessuna conoscenza di cosa cost_fn faccia
    internamente: il "significato fisico" di J (fan, propeller, una loro
    combinazione pesata, o altro ancora) è deciso interamente da quale
    cost_fn passi qui, non da questo modulo.

    cost_kwargs: passati così come sono a cost_fn (es. per combined_cost:
    {"fan_weight": 2.0, "propeller_weight": 1.0}).

    penalty_on_error: valore restituito se la valutazione del modello
    solleva un'eccezione per un dato theta (es. bounds troppo larghi che
    portano a punti fisicamente degeneri) - grande ma finito, per non far
    bloccare l'ottimizzatore.
    """
    cost_kwargs = cost_kwargs or {}

    def J(theta):
        try:
            tech, wtt = theta_to_tech_wtt(theta, param_names, base_tech, base_wtt)
            total, _ = cost_fn(tech, wtt, **cost_kwargs)
            if not np.isfinite(total):
                return penalty_on_error
            return total
        except Exception:
            return penalty_on_error

    return J


# ---------------------------------------------------------------------
# 3. Punti iniziali multipli (Latin Hypercube) e loop di calibrazione
# ---------------------------------------------------------------------

def latin_hypercube_starts(bounds: dict, n_starts: int, seed: Optional[int] = None):
    """n_starts punti iniziali via Latin Hypercube Sampling entro bounds
    (par. 5.1: "non utilizzare un solo punto iniziale" - LHS invece di
    random puro per una copertura più uniforme dello spazio dei parametri).

    Ritorna (param_names, starts) con starts di shape (n_starts, n_theta),
    stesso ordine di bounds (un dict: in Python 3.7+ preserva l'ordine di
    inserimento).
    """
    param_names = list(bounds.keys())
    lo = np.array([bounds[p][0] for p in param_names])
    hi = np.array([bounds[p][1] for p in param_names])
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    unit_sample = sampler.random(n=n_starts)
    starts = qmc.scale(unit_sample, lo, hi)
    return param_names, starts


@dataclass
class CalibrationRun:
    """Esito di un singolo tentativo di ottimizzazione (un punto iniziale,
    o l'unica run di differential_evolution)."""
    method: str
    x0: dict = field(repr=False)
    theta_star: dict
    cost: float
    success: bool
    n_eval: int


def run_deterministic_calibration(J: Callable, param_names, bounds: Optional[dict] = None,
                                   method: str = "differential_evolution",
                                   n_starts: int = 8, n_de_runs: int = 3,
                                   nelder_mead_kwargs: Optional[dict] = None,
                                   de_kwargs: Optional[dict] = None,
                                   base_tech: Optional[TechAssumptions] = None,
                                   base_wtt: Optional[WellToTankEfficiencies] = None,
                                   seed: int = 0, verbose: bool = True):
    """Risolve theta* = argmin_theta J(theta) (par. 5.1).

    J: la funzione di costo da minimizzare, R^n_theta -> R, tipicamente
    costruita con make_objective(cost_fn, param_names, ...) a partire da
    una cost_fn di cnav.calibration.cost_functions (fan_cost, propeller_cost,
    combined_cost, ...) - ma questo modulo accetta qualunque callable con
    quella firma, non ha nessuna conoscenza di cosa J rappresenti
    fisicamente. base_tech/base_wtt qui sotto servono solo per calcolare
    bounds di default quando bounds=None (default_bounds ha bisogno dei
    valori nominali) - se costruisci J con base_tech/base_wtt diversi,
    passa qui gli stessi per coerenza.

    method controlla quale/i metodo/i usare, tra quelli elencati al par.
    5.1 ("Least-squares; Nelder-Mead; differential evolution; CMA-ES;
    gradient-based, se il modello è sufficientemente regolare" - qui
    coperti Nelder-Mead e differential_evolution; vedi il docstring del
    modulo per il perché gradient-based è lasciato fuori per ora):

      - "differential_evolution" (default): SOLO differential_evolution,
        ripetuta n_de_runs volte con seed diversi (default 3). DE è già
        di per sé un metodo globale, popolazione-based: qui "più punti
        iniziali diversi" (par. 5.1) si traduce nel ripetere l'intera
        ottimizzazione con popolazioni iniziali diverse, non nel dare a
        una singola DE punti di partenza multipli (non avrebbe senso,
        DE non prende un x0).
      - "nelder_mead": SOLO Nelder-Mead, da n_starts punti iniziali
        diversi (Latin Hypercube entro bounds).
      - "both": entrambi (n_de_runs DE + n_starts Nelder-Mead).

    Ritorna (best_run, df_runs):
      - best_run: il CalibrationRun con il costo più basso;
      - df_runs: un DataFrame con un tentativo per riga (method, cost,
        success, n_eval, theta_<nome> per ciascun parametro), ordinato
        per costo crescente - da ispezionare per la dispersione dei minimi
        trovati (preliminare al par. 5.2, vedi summarize_multiple_minima).

    Costo computazionale: ogni valutazione di J(theta) chiama il modello
    completo (via la cost_fn con cui hai costruito J) e costa
    indicativamente 0.3-0.6s (di più se J combina più configurazioni, es.
    combined_cost chiama sia fan_cost che propeller_cost). Una singola
    differential_evolution con i default sotto (maxiter=60, popsize=12,
    per n_theta=6 -> popolazione iniziale 72) valuta tipicamente
    qualche migliaio di punti: anche solo UNA run può richiedere
    10-20 minuti, quindi n_de_runs=3 di default può richiedere
    mezz'ora abbondante. Per un primo giro di prova riduci
    de_kwargs={"maxiter": 15, "popsize": 8} e n_de_runs=1; una volta
    verificato che il flusso funziona, alza i budget per la calibrazione
    "vera" da riportare in tesi (idealmente girata offline/in background,
    non in una sessione interattiva).
    """
    if method not in ("differential_evolution", "nelder_mead", "both"):
        raise ValueError("method deve essere 'differential_evolution', 'nelder_mead' o 'both'")

    if bounds is None:
        bounds = default_bounds(param_names, tech=base_tech, wtt=base_wtt)
    bounds = {p: bounds[p] for p in param_names}  # forza l'ordine di param_names
    bounds_list = [bounds[p] for p in param_names]

    runs = []

    if method in ("nelder_mead", "both"):
        names, starts = latin_hypercube_starts(bounds, n_starts, seed=seed)
        assert names == param_names

        nm_options = dict(maxiter=500, xatol=1e-6, fatol=1e-8)
        if nelder_mead_kwargs:
            nm_options.update(nelder_mead_kwargs)

        for i, x0 in enumerate(starts):
            res = minimize(J, x0, method="Nelder-Mead", bounds=bounds_list, options=nm_options)
            runs.append(CalibrationRun(
                method=f"Nelder-Mead (start {i})",
                x0=dict(zip(param_names, x0)),
                theta_star=dict(zip(param_names, res.x)),
                cost=float(res.fun), success=bool(res.success), n_eval=int(res.nfev),
            ))
            if verbose:
                print(f"  [{i + 1}/{n_starts}] Nelder-Mead   J* = {res.fun:.6g}  "
                      f"(success={res.success}, nfev={res.nfev})")

    if method in ("differential_evolution", "both"):
        # polish=False: il polish di default di differential_evolution è un
        # L-BFGS-B locale finale (gradiente per differenze finite). J(theta)
        # non è garantita liscia (vedi docstring del modulo: AircraftSizer ha
        # una condizione di convergenza booleana, con penalty discontinuo se
        # non converge), e in pratica il polish può restare "impuntato" a
        # cercare un gradiente su un tratto piatto/discontinuo, moltiplicando
        # per decine il numero di valutazioni senza migliorare il risultato.
        # workers=1: J chiude su fan_cost/propeller_cost (closure locale),
        # non è picklabile per il multiprocessing di workers>1/-1 (se vuoi
        # parallelizzare davvero, serve rendere J una funzione di modulo,
        # non una closure - vedi la nota in fondo al modulo).
        de_options = dict(maxiter=60, popsize=12, tol=1e-7, workers=1, polish=False)
        if de_kwargs:
            de_options.update(de_kwargs)

        for run_idx in range(n_de_runs):
            de_seed = seed + run_idx
            res_de = differential_evolution(J, bounds_list, seed=de_seed, **de_options)
            runs.append(CalibrationRun(
                method=f"differential_evolution (seed {de_seed})", x0={},
                theta_star=dict(zip(param_names, res_de.x)),
                cost=float(res_de.fun), success=bool(res_de.success), n_eval=int(res_de.nfev),
            ))
            if verbose:
                print(f"  [{run_idx + 1}/{n_de_runs}] differential_evolution "
                      f"(seed={de_seed})  J* = {res_de.fun:.6g}  "
                      f"(success={res_de.success}, nfev={res_de.nfev})")

    df_runs = pd.DataFrame([
        {"method": r.method, "cost": r.cost, "success": r.success, "n_eval": r.n_eval,
         **{f"theta_{k}": v for k, v in r.theta_star.items()}}
        for r in runs
    ]).sort_values("cost").reset_index(drop=True)

    best_run = min(runs, key=lambda r: r.cost)
    return best_run, df_runs


# ---------------------------------------------------------------------
# 4. Diagnostica preliminare sui minimi multipli (anticipa il par. 5.2)
# ---------------------------------------------------------------------

def summarize_multiple_minima(df_runs: pd.DataFrame, rel_tol: float = 0.05):
    """A partire da df_runs (output di run_deterministic_calibration),
    isola i run con costo entro rel_tol (default 5%) dal minimo globale
    trovato e verifica se, tra questi, il theta* trovato varia in modo
    apprezzabile (deviazione standard relativa > 5% su almeno un
    parametro) - un primo indizio di equifinality (par. 5.2), da
    approfondire poi con un'analisi dedicata (non sostituisce quella).

    Ritorna (near_best, equifinality_suspected): near_best è il
    sotto-DataFrame dei run vicini al minimo, equifinality_suspected è un
    booleano.
    """
    if df_runs.empty:
        return df_runs, False
    best_cost = df_runs["cost"].min()
    threshold = best_cost * (1 + rel_tol) if best_cost > 0 else best_cost + rel_tol
    near_best = df_runs[df_runs["cost"] <= threshold].copy()

    theta_cols = [c for c in df_runs.columns if c.startswith("theta_")]
    if len(near_best) <= 1 or not theta_cols:
        return near_best, False

    means = near_best[theta_cols].mean().abs().replace(0, np.nan)
    rel_spread = near_best[theta_cols].std() / means
    equifinality_suspected = bool((rel_spread > 0.05).any())
    return near_best, equifinality_suspected
