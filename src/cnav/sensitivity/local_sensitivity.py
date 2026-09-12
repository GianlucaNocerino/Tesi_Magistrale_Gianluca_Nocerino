import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..calibration.cost_functions import max_feasible_range_nmi
from ..model.constants import TechAssumptions, WellToTankEfficiencies
from ..model.energy_intensity import most_efficient_system
from ..model.mission import Mission
from ..model.propulsion_systems import build_default_systems
from ..model.well_to_tank import build_energy_carriers
from .sensitivity_analysis_tools import (
    PARAMETER_REGISTRY,
    _compute_param_elasticities,
    _elasticity,
    get_flat_nominal_params,
    plot_tornado_from_sweep,
)

"""
Analisi di sensibilità locale:

Per ciascun parametro theta_i, l'indice di sensibilità locale è

    S_loc_i = (theta_i / y) * (dy / dtheta_i)

approssimato con differenze finite centrate:

    S_loc_i ~= (theta_i / y) * (y(theta_i+Delta_i) - y(theta_i-Delta_i)) / (2*Delta_i)

con perturbazioni relative Delta_i/theta_i inizialmente nell'ordine
1%-5%, verificando che il risultato non dipenda in modo significativo
dalla dimensione della perturbazione. L'analisi va ripetuta per più
condizioni operative rappresentative:
    - short-range / low-speed
    - short-range / high-speed
    - medium-range
    - long-range
    - punti prossimi ai confini tra tecnologie

Questo modulo non reimplementa il calcolo delle elasticità: lo riusa
da sensitivity_analysis_tools.py (evaluate_model, _compute_param_elasticities,
PARAMETER_REGISTRY) e struttura la procedura richiesta dalla
consegna: condizioni operative standard, verifica di robustezza rispetto
alla perturbazione, individuazione dei confini tra tecnologie.

Il risultato principale (run_local_sensitivity_report) è tabellare
(due DataFrame pandas), non grafico. Per la visualizzazione si possono
riusare gli strumenti già presenti in sensitivity_analysis_tools.py:
df_elasticities ha lo stesso schema di colonne prodotto da
run_sensitivity_sweep (range_nmi, cruise_speed_kt, propulsor,
system_name, param_name, elasticity), quindi plot_tornado_from_sweep()
funziona già direttamente su di esso — vedi anche il wrapper
plot_condition_tornado() più sotto, che risolve automaticamente
range/velocità/propulsore a partire dall'etichetta della condizione.
"""

DEFAULT_SYSTEM_NAMES = [
    "Battery-electric", "Hydrogen fuel cell", "Hydrogen combustion", "e-SAF combustion",
]

# ---------------------------------------------------------------------
# Condizioni operative standard richieste dalla consegna.
#
#    Scelte indicative (range/velocità), coerenti con i punti già usati
#    altrove nel progetto (paper: fan @450kt, propeller @250kt; range
#    10-10000 nmi), adattabili liberamente. "low/high speed" qui distingue 
#    elica (più lenta) da fan (più veloce) a parità di range corto, dato 
#    che nel modello la velocità di crociera è legata al tipo di propulsore.
# ---------------------------------------------------------------------
OPERATING_CONDITIONS = {
    "short_range_low_speed_propeller":  Mission(range_nmi=100.0,  cruise_speed_kt=200.0, propulsor="propeller"),
    "short_range_low_speed_fan":        Mission(range_nmi=100.0,  cruise_speed_kt=350.0, propulsor="fan"),
    "short_range_high_speed_propeller": Mission(range_nmi=100.0,  cruise_speed_kt=300.0, propulsor="propeller"),
    "short_range_high_speed_fan":       Mission(range_nmi=100.0,  cruise_speed_kt=450.0, propulsor="fan"),
    "very_short_range_propeller":       Mission(range_nmi=30.0,   cruise_speed_kt=250.0, propulsor="propeller"),
    "very_short_range_fan":             Mission(range_nmi=15.0,   cruise_speed_kt=450.0, propulsor="fan"),
    "medium_range_propeller":           Mission(range_nmi=1500.0, cruise_speed_kt=250.0, propulsor="propeller"),
    "medium_range_fan":                 Mission(range_nmi=1500.0, cruise_speed_kt=350.0, propulsor="fan"),
    "long_range_propeller":             Mission(range_nmi=6000.0, cruise_speed_kt=250.0, propulsor="propeller"),
    "long_range_fan":                   Mission(range_nmi=6000.0, cruise_speed_kt=450.0, propulsor="fan"),
}

# ---------------------------------------------------------------------
# Verifica di robustezza rispetto alla dimensione della perturbazione
# ---------------------------------------------------------------------

def perturbation_robustness(mission: Mission, system_name: str, param_subset=None,
                             perturbations=(0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """Calcola S_loc per lo stesso punto operativo/sistema a più
    dimensioni di perturbazione relativa (default 1%-5%, come da
    consegna). Ritorna un DataFrame tidy (perturbation, param_name,
    elasticity), input di flag_unstable_parameters
    """
    rows = []
    for pert in perturbations:
        _, results = _compute_param_elasticities(mission, system_name, pert, param_subset)
        for param_name, elasticity in results.items():
            rows.append({"perturbation": pert, "param_name": param_name, "elasticity": elasticity})
    return pd.DataFrame(rows)


def flag_unstable_parameters(df_robustness: pd.DataFrame, rel_tol: float = 0.10) -> pd.DataFrame:
    """A partire dal DataFrame di perturbation_robustness, per ciascun
    parametro calcola l'escursione relativa di S_loc tra le perturbazioni
    testate (max-min, rispetto alla mediana) e segnala 'stable=False' se
    supera rel_tol (default 10%): stiamo verificando che il risultato non dipenda in modo
    significativo dalla dimensione della perturbazione
    """
    if df_robustness.empty:
        return pd.DataFrame(columns=["param_name", "rel_spread", "stable"])

    def _rel_spread(group):
        elasticities = group["elasticity"]
        med = elasticities.median()
        abs_spread = elasticities.max() - elasticities.min()
        if abs(med) < 1e-8:
            # elasticità sostanzialmente nulla (parametro irrilevante per
            # questo sistema/punto): stabile se anche l'escursione assoluta
            # è trascurabile, altrimenti non è ben definito un rapporto
            # relativo utile, quindi segnaliamo comunque instabile
            return 0.0 if abs_spread < 1e-6 else np.inf
        return abs_spread / abs(med)

    spread = (df_robustness.groupby("param_name").apply(_rel_spread)
              .reset_index(name="rel_spread"))
    spread["stable"] = spread["rel_spread"] <= rel_tol
    return spread.sort_values("rel_spread", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Elasticità del range massimo fattibile (solo Battery-electric)
#
#    calibration.max_feasible_range_nmi non è una funzione di (mission,
#    system_name) come compute_intensity: dipende solo da (propulsor,
#    speed_kt) e dai parametri. Stessa formula a differenze centrate di
#    _compute_param_elasticities, ma con y = range massimo invece di
#    y = intensity.
# ---------------------------------------------------------------------

def _tech_wtt_from_flat(flat_params: dict):
    """Ricostruisce (tech, wtt) da un dict piatto di parametri — stessa
    logica di evaluate_model in sensitivity_analysis_tools.py, ma
    restituisce gli oggetti invece di valutare il modello, qui serve per
    poter chiamare max_feasible_range_nmi(tech, wtt, ...)"""
    default_tech = TechAssumptions()
    default_wtt = WellToTankEfficiencies()
    tech_kwargs, wtt_kwargs = {}, {}
    for key, value in flat_params.items():
        if hasattr(default_tech, key):
            tech_kwargs[key] = value
        elif hasattr(default_wtt, key):
            wtt_kwargs[key] = value
    return default_tech.with_changes(**tech_kwargs), default_wtt.with_changes(**wtt_kwargs)


def _compute_range_elasticities(propulsor: str, speed_kt: float, perturbation: float = 0.01,
                                 param_subset=None, tol_nmi: float = 2.0, max_bisect_iter: int = 30):
    """Elasticità di max_feasible_range_nmi (Battery-electric) rispetto
    a ciascun parametro, a un dato (propulsor, speed_kt). Stessa firma
    di ritorno di _compute_param_elasticities: (y_nom, {param: elasticità}).

    tol_nmi/max_bisect_iter sono più permissivi di quelli di default di
    max_feasible_range_nmi (0.5 nmi / 60 iter, pensati per la
    calibrazione): qui ogni valutazione di S_loc richiede ~2*n_param
    bisezioni indipendenti (una per lato di perturbazione, per ogni
    parametro), quindi il costo si accumula in fretta — 2 nmi di
    tolleranza sono più che sufficienti per un indice di elasticità
    """
    base_params = get_flat_nominal_params()
    tech0, wtt0 = _tech_wtt_from_flat(base_params)
    y_nom = max_feasible_range_nmi(tech0, wtt0, propulsor, speed_kt, tol_nmi=tol_nmi,
                                    max_bisect_iter=max_bisect_iter)

    if np.isnan(y_nom) or y_nom == 0.0:
        return y_nom, {}

    results = {}
    for param_name, param_data in PARAMETER_REGISTRY.items():
        if param_subset is not None and param_name not in param_subset:
            continue

        nominal_val = param_data["nominal"]
        if nominal_val == 0.0 or np.isnan(nominal_val):
            continue

        delta_x = nominal_val * perturbation
        params_plus = base_params.copy(); params_plus[param_name] = nominal_val + delta_x
        params_minus = base_params.copy(); params_minus[param_name] = nominal_val - delta_x

        tech_plus, wtt_plus = _tech_wtt_from_flat(params_plus)
        tech_minus, wtt_minus = _tech_wtt_from_flat(params_minus)

        y_plus = max_feasible_range_nmi(tech_plus, wtt_plus, propulsor, speed_kt,
                                         tol_nmi=tol_nmi, max_bisect_iter=max_bisect_iter)
        y_minus = max_feasible_range_nmi(tech_minus, wtt_minus, propulsor, speed_kt,
                                          tol_nmi=tol_nmi, max_bisect_iter=max_bisect_iter)

        s_index = _elasticity(y_nom, y_plus, y_minus, perturbation)
        if not np.isnan(s_index):
            results[param_name] = s_index

    return y_nom, results


def range_perturbation_robustness(propulsor: str, speed_kt: float, param_subset=None,
                                   perturbations=(0.01, 0.02, 0.03, 0.05),
                                   tol_nmi: float = 2.0, max_bisect_iter: int = 30) -> pd.DataFrame:
    """Equivalente di perturbation_robustness, ma per max_feasible_range_nmi
    invece dell'intensity"""
    rows = []
    for pert in perturbations:
        _, results = _compute_range_elasticities(propulsor, speed_kt, pert, param_subset,
                                                   tol_nmi=tol_nmi, max_bisect_iter=max_bisect_iter)
        for param_name, elasticity in results.items():
            rows.append({"perturbation": pert, "param_name": param_name, "elasticity": elasticity})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Punti prossimi ai confini tra tecnologie
# ---------------------------------------------------------------------

def find_technology_boundaries(propulsor: str, speed_kt: float,
                                tech: TechAssumptions = None, wtt: WellToTankEfficiencies = None,
                                range_grid_nmi=None, tol_nmi: float = 1.0,
                                max_bisect_iter: int = 40) -> list:
    """Individua, a velocità fissata, i Range in cui il sistema più
    efficiente (most_efficient_system) cambia, per bisezione sul cambio
    di "best system" tra due punti consecutivi di una griglia grezza di
    partenza. Ritorna una lista di dict {range_nmi, system_before, system_after}
    """
    tech = tech or TechAssumptions()
    wtt = wtt or WellToTankEfficiencies()
    if range_grid_nmi is None:
        range_grid_nmi = np.geomspace(10, 10_000, 60)

    systems = build_default_systems(build_energy_carriers(wtt))

    def best_at(r_nmi):
        mission = Mission(range_nmi=float(r_nmi), cruise_speed_kt=speed_kt, propulsor=propulsor)
        return most_efficient_system(mission, systems, tech)["best"]

    best_names = [best_at(r) for r in range_grid_nmi]

    boundaries = []
    for i in range(1, len(range_grid_nmi)):
        name_before, name_after = best_names[i - 1], best_names[i]
        if name_before is None or name_after is None or name_before == name_after:
            continue
        lo, hi = float(range_grid_nmi[i - 1]), float(range_grid_nmi[i])
        for _ in range(max_bisect_iter):
            mid = 0.5 * (lo + hi)
            if best_at(mid) == name_before:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol_nmi:
                break
        boundaries.append({
            "range_nmi": 0.5 * (lo + hi),
            "system_before": name_before,
            "system_after": name_after,
        })
    return boundaries


def boundary_conditions(propulsor: str, speed_kt: float,
                         tech: TechAssumptions = None, wtt: WellToTankEfficiencies = None,
                         range_grid_nmi=None) -> dict:
    """Converte i confini trovati da find_technology_boundaries in un
    dict {etichetta: Mission}, nello stesso formato di OPERATING_CONDITIONS"""
    boundaries = find_technology_boundaries(propulsor, speed_kt, tech, wtt, range_grid_nmi)
    out = {}
    for b in boundaries:
        label = f"boundary_{propulsor}_{b['system_before']}_to_{b['system_after']}".replace(" ", "_")
        out[label] = Mission(range_nmi=b["range_nmi"], cruise_speed_kt=speed_kt, propulsor=propulsor)
    return out


# ---------------------------------------------------------------------
# Procedura completa
# ---------------------------------------------------------------------

def run_local_sensitivity_report(system_names=None,
                                  tech: TechAssumptions = None, wtt: WellToTankEfficiencies = None,
                                  reference_perturbation: float = 0.02,
                                  robustness_perturbations=(0.01, 0.02, 0.03, 0.05),
                                  extra_conditions: dict = None,
                                  include_boundaries: bool = True,
                                  boundary_propulsor_speed=(("fan", 450.0), ("propeller", 250.0)),
                                  stability_tol: float = 0.10,
                                  include_battery_range: bool = True,
                                  range_param_subset=None,
                                  compute_range_robustness: bool = False,
                                  range_robustness_perturbations=None,
                                  verbose: bool = True):
    """
    1) unisce le condizioni operative standard (OPERATING_CONDITIONS) con
       eventuali extra_conditions dell'utente e, se include_boundaries,
       con i punti di confine tra tecnologie trovati per ciascuna coppia
       (propulsor, speed_kt) in boundary_propulsor_speed;
    2) per ogni condizione e ogni sistema, calcola S_loc di tutti i
       parametri a reference_perturbation;
    3) per ogni condizione e ogni sistema, verifica la stabilità di
       S_loc su robustness_perturbations (tipicamente 1%-5%);
    4) se include_battery_range, aggiunge anche S_loc del range massimo
       fattibile per Battery-electric (vedi _compute_range_elasticities)

    Costo: il termine sul range massimo richiede una bisezione per ogni
    lato di ogni parametro perturbato — molto più costoso del termine
    sull'intensity (indicativamente, ~10-15s per punto operativo per la
    sola elasticità di riferimento, con tutti i ~40 parametri; la
    verifica di robustezza lo moltiplica per il numero di perturbazioni
    testate). Le condizioni/confini che condividono lo stesso
    (propulsor, cruise_speed_kt) vengono calcolati una sola volta
    (cache interna). Per questo compute_range_robustness è False di
    default (si tiene solo l'elasticità di riferimento sul range,
    veloce); attivalo esplicitamente quando serve davvero. Altri modi
    per velocizzare ulteriormente: range_param_subset per limitare i
    parametri testati sul range, o boundary_propulsor_speed=() per
    escludere i confini dal calcolo del range.

    Ritorna (df_elasticities, df_robustness):
    - df_elasticities: una riga per (condition, system_name, output,
      param_name), con l'elasticità di riferimento. 'output' vale
      "intensity" per tutte le righe, più "max_range_nmi" per
      Battery-electric (elasticità del range massimo fattibile,
      indipendente dal range_nmi della condizione: dipende solo da
      propulsor e cruise_speed_kt — range_nmi vale -1.0, sentinella,
      per queste righe).
    - df_robustness: stessa struttura, con l'escursione relativa tra
      perturbazioni e il flag 'stable'.
    """
    if system_names is None:
        system_names = DEFAULT_SYSTEM_NAMES
    tech = tech or TechAssumptions()
    wtt = wtt or WellToTankEfficiencies()
    if range_robustness_perturbations is None:
        range_robustness_perturbations = robustness_perturbations

    conditions = dict(OPERATING_CONDITIONS)
    if extra_conditions:
        conditions.update(extra_conditions)
    if include_boundaries:
        for propulsor, speed_kt in boundary_propulsor_speed:
            conditions.update(boundary_conditions(propulsor, speed_kt, tech, wtt))

    elasticity_rows = []
    robustness_rows = []
    range_cache = {}  # (propulsor, speed_kt) -> (y_nom, elasticities, flags_df_o_None)

    for label, mission in conditions.items():
        # --- elasticità del range massimo fattibile (solo Battery-electric) ---
        if include_battery_range and "Battery-electric" in system_names:
            key = (mission.propulsor, mission.cruise_speed_kt)
            if key not in range_cache:
                y_nom_range, range_results = _compute_range_elasticities(
                    *key, reference_perturbation, param_subset=range_param_subset)
                flags = None
                if not np.isnan(y_nom_range) and compute_range_robustness:
                    df_pert_range = range_perturbation_robustness(
                        *key, param_subset=range_param_subset,
                        perturbations=range_robustness_perturbations)
                    if not df_pert_range.empty:
                        flags = flag_unstable_parameters(df_pert_range, rel_tol=stability_tol)
                range_cache[key] = (y_nom_range, range_results, flags)

            y_nom_range, range_results, flags = range_cache[key]
            if np.isnan(y_nom_range):
                if verbose:
                    print(f"[SKIP] {label}: range massimo fattibile non definito "
                          f"({mission.propulsor} @ {mission.cruise_speed_kt:.0f} kt)")
            else:
                for param_name, elasticity in range_results.items():
                    elasticity_rows.append({
                        # range_nmi=-1.0 e' un valore sentinella (non un
                        # range fisico): l'elasticita' del range massimo
                        # fattibile non dipende dal range_nmi della
                        # condizione, solo da propulsor/cruise_speed_kt.
                        # Un numero (non NaN) e' necessario perche'
                        # plot_condition_tornado/plot_tornado_from_sweep
                        # filtrano per uguaglianza, e NaN == NaN e' sempre
                        # False in pandas.
                        "condition": label, "range_nmi": -1.0,
                        "cruise_speed_kt": mission.cruise_speed_kt, "propulsor": mission.propulsor,
                        "system_name": "Battery-electric", "output": "max_range_nmi",
                        "category": PARAMETER_REGISTRY[param_name]["category"],
                        "param_name": param_name, "elasticity": elasticity,
                    })
                if flags is not None:
                    flags = flags.copy()
                    flags["condition"] = label
                    flags["system_name"] = "Battery-electric"
                    flags["output"] = "max_range_nmi"
                    robustness_rows.append(flags)

        # --- elasticità dell'intensity, per ogni sistema ---
        for system_name in system_names:
            y_nom, results = _compute_param_elasticities(mission, system_name, reference_perturbation)
            if np.isnan(y_nom):
                if verbose:
                    print(f"[SKIP] {label}: {system_name} non converge a "
                          f"({mission.range_nmi:.0f} nmi, {mission.cruise_speed_kt:.0f} kt, {mission.propulsor})")
                continue

            for param_name, elasticity in results.items():
                elasticity_rows.append({
                    "condition": label, "range_nmi": mission.range_nmi,
                    "cruise_speed_kt": mission.cruise_speed_kt, "propulsor": mission.propulsor,
                    "system_name": system_name, "output": "intensity",
                    "category": PARAMETER_REGISTRY[param_name]["category"],
                    "param_name": param_name, "elasticity": elasticity,
                })

            df_pert = perturbation_robustness(mission, system_name, perturbations=robustness_perturbations)
            if not df_pert.empty:
                flags = flag_unstable_parameters(df_pert, rel_tol=stability_tol)
                flags["condition"] = label
                flags["system_name"] = system_name
                flags["output"] = "intensity"
                robustness_rows.append(flags)

    df_elasticities = pd.DataFrame(elasticity_rows)
    df_robustness = (pd.concat(robustness_rows, ignore_index=True)
                      if robustness_rows else pd.DataFrame(
                          columns=["param_name", "rel_spread", "stable", "condition", "system_name", "output"]))
    return df_elasticities, df_robustness


# ---------------------------------------------------------------------
# Visualizzazione: riuso di plot_tornado_from_sweep (sensitivity_analysis_tools.py)
# ---------------------------------------------------------------------

def plot_condition_tornado(df_elasticities: pd.DataFrame, condition: str, system_name: str,
                            output: str = "intensity", top_n=None, threshold: float = 1e-4):
    """Tornado plot per una condizione/sistema, leggendo i dati già
    calcolati in df_elasticities (output di run_local_sensitivity_report).
    Wrapper su plot_tornado_from_sweep, che si
    aspetta (range_nmi, cruise_speed_kt, propulsor, system_name)
    anziché l'etichetta testuale 'condition': qui li recupera in
    automatico dalla prima riga che corrisponde a condition/system_name/output.

    output: "intensity" (default) o "max_range_nmi" (solo Battery-electric).
    threshold: mostra solo i parametri con |elasticità| > threshold
    (criterio principale di selezione, vedi plot_tornado_from_sweep).
    """
    subset = df_elasticities[
        (df_elasticities["condition"] == condition) &
        (df_elasticities["system_name"] == system_name) &
        (df_elasticities["output"] == output)
    ]
    if subset.empty:
        print(f"Nessun dato per condition={condition!r}, system_name={system_name!r}, output={output!r}.")
        return

    row0 = subset.iloc[0]
    # per output="max_range_nmi", range_nmi=-1.0 e' il valore sentinella
    # scritto da run_local_sensitivity_report (vedi commento lì): lo
    # ripassiamo cosi' com'e' al filtro per uguaglianza di plot_tornado_from_sweep
    range_nmi = row0["range_nmi"]

    # plot_tornado_from_sweep filtra solo su (range_nmi, cruise_speed_kt,
    # propulsor, system_name), non su 'condition': per output=
    # "max_range_nmi" più condizioni possono condividere lo stesso
    # (propulsor, cruise_speed_kt) (l'elasticità del range massimo non
    # dipende dal range_nmi della condizione), producendo righe duplicate
    # nel filtro. Le togliamo qui prima di passare i dati al plot.
    plot_df = df_elasticities[df_elasticities["output"] == output].drop_duplicates(
        subset=["range_nmi", "cruise_speed_kt", "propulsor", "system_name", "param_name"])

    # titolo esplicito sull'output: plot_tornado_from_sweep di suo non lo
    # sa (il suo titolo di default non menziona quale grandezza è stata
    # perturbata), e per max_range_nmi mostrerebbe anche il fuorviante
    # "range_nmi=-1.0" (il sentinella) se non lo sostituissimo qui.
    if output == "max_range_nmi":
        title = (f'Tornado Plot - {system_name} ({row0["propulsor"]})\n'
                  f'Range massimo fattibile — {row0["cruise_speed_kt"]:.0f} kt\n[{condition}]')
    else:
        title = (f'Tornado Plot - {system_name} ({row0["propulsor"]}) — Electricity Intensity\n'
                  f'{row0["range_nmi"]:.0f} nmi, {row0["cruise_speed_kt"]:.0f} kt\n[{condition}]')

    plot_tornado_from_sweep(
        plot_df, range_nmi=range_nmi, cruise_speed_kt=row0["cruise_speed_kt"],
        propulsor=row0["propulsor"], system_name=system_name, top_n=top_n, threshold=threshold,
        title=title,
    )


def plot_all_condition_tornados(df_elasticities: pd.DataFrame, top_n=None, threshold: float = 1e-4):
    """Genera un tornado plot per ogni combinazione (condition,
    system_name, output) presente in df_elasticities — comodo per
    rivedere tutte le condizioni in un colpo solo, invece
    di richiamare plot_condition_tornado singolarmente per ciascuna.

    threshold: mostra solo i parametri con |elasticità| > threshold in
    ciascun tornado (vedi plot_tornado_from_sweep).

    Attenzione: con 8 condizioni standard + confini, 4 sistemi e 2
    output (intensity + max_range_nmi per Battery-electric), il numero
    di figure generate può essere alto (tipicamente sull'ordine di
    30-40) — ciascuna si apre come figura separata"""
    combos = df_elasticities[["condition", "system_name", "output"]].drop_duplicates()
    for _, row in combos.iterrows():
        plot_condition_tornado(df_elasticities, condition=row["condition"],
                                system_name=row["system_name"], output=row["output"],
                                top_n=top_n, threshold=threshold)

# ---------------------------------------------------------------------
# Indice di elasticità medio dei parametri rilevanti, per sistema
#
#    "Rilevante" = stesso criterio di selezione usato dai tornado plot
#    (plot_tornado_from_sweep/plot_condition_tornado): |elasticità| >
#    threshold. Qui non si ricalcola nulla: si riusa df_elasticities
#    (output di run_local_sensitivity_report) e si applica lo stesso
#    filtro, poi si fa la media SOLO sulle condizioni in cui il
#    parametro ha effettivamente superato la soglia (cioè sulle
#    condizioni in cui sarebbe comparso nel tornado plot). Le condizioni
#    in cui il parametro non è rilevante non entrano nella media
#
# ---------------------------------------------------------------------

def average_relevant_elasticities(df_elasticities: pd.DataFrame, threshold: float,
                                   group_cols=("system_name", "propulsor", "output")) -> pd.DataFrame:
    """Per ogni combinazione di group_cols (default: sistema, propulsore,
    output) e per ogni parametro, calcola l'indice di elasticità medio
    S_loc sulle sole condizioni operative in cui |S_loc| > threshold
    (cioè le condizioni in cui il parametro comparirebbe nel tornado
    plot corrispondente).

    threshold non ha un default apposta: deve coincidere con quello
    usato per generare i tornado plot che si vogliono riassumere (es.
    execution/03_sensibilita_locale/local_sensitivity_report.py chiama
    plot_all_condition_tornados(df_elasticities, threshold=0.05), se
    qui si passasse un valore diverso, "parametro rilevante" avrebbe un
    significato diverso da quello dei tornado plot effettivamente
    generati, ed è esattamente l'ambiguità da evitare.

    Ritorna un DataFrame con una riga per (system_name, propulsor,
    output, param_name, category):
    - mean_elasticity: media di S_loc (con segno) sulle condizioni rilevanti
    - std_elasticity: deviazione standard di S_loc sulle stesse condizioni
      (NaN se n_relevant == 1); utile per capire se il segno/l'ordine di
      grandezza è stabile o cambia tra le condizioni (es. vicino a un
      confine tra tecnologie)
    - mean_abs_elasticity: media di |S_loc|, usata per l'ordinamento -
      non si vuole che un parametro che oscilla di segno tra condizioni
      diverse (media con segno vicina a 0) sembri poco influente
    - n_relevant: in quante condizioni il parametro è risultato rilevante
    - n_conditions_total: quante condizioni sono state valutate in totale
      per quella combinazione (n_relevant/n_conditions_total dà una
      misura di quanto "sistematicamente" rilevante sia il parametro,
      non solo isolato in un punto)
    """
    group_cols = list(group_cols)
    total_counts = (df_elasticities
                     .drop_duplicates(subset=group_cols + ["condition", "system_name"])
                     .groupby(group_cols)
                     .size()
                     .rename("n_conditions_total")
                     .reset_index())

    relevant = df_elasticities[df_elasticities["elasticity"].abs() > threshold]
    if relevant.empty:
        return pd.DataFrame(columns=group_cols + ["param_name", "category", "mean_elasticity",
                                                    "std_elasticity", "mean_abs_elasticity",
                                                    "std_abs_elasticity", "n_relevant",
                                                    "n_conditions_total"])

    grouped = (relevant.groupby(group_cols + ["param_name", "category"])
               .agg(mean_elasticity=("elasticity", "mean"),
                    std_elasticity=("elasticity", "std"),
                    mean_abs_elasticity=("elasticity", lambda s: s.abs().mean()),
                    std_abs_elasticity=("elasticity", lambda s: s.abs().std()),
                    n_relevant=("elasticity", "count"))
               .reset_index())
    grouped = grouped.merge(total_counts, on=group_cols, how="left")
    grouped = grouped.sort_values(group_cols + ["mean_abs_elasticity"],
                                   ascending=[True] * len(group_cols) + [False])
    return grouped.reset_index(drop=True)


def plot_average_elasticity(df_avg: pd.DataFrame, system_name: str, propulsor: str,
                             output: str = "intensity", top_n=None, min_n_relevant: int = 2,
                             title: str = None, reference_line: float = None):
    """Tornado-style plot dell'indice di elasticità medio (da
    average_relevant_elasticities) per un singolo (system_name,
    propulsor, output), es. plot_average_elasticity(df_avg,
    "Battery-electric", "fan").

    Le barre di errore mostrano ±1 deviazione standard tra le condizioni
    su cui è stata fatta la media (solo se n_relevant > 1). L'etichetta
    di ogni parametro riporta anche "n=n_relevant/n_conditions_total",
    per distinguere a colpo d'occhio un parametro rilevante quasi
    ovunque da uno rilevante solo in una condizione isolata.

    min_n_relevant: mostra solo i parametri rilevanti in almeno questo
    numero di condizioni (default 2, non 1: con n_relevant=1 la "media"
    coincide col valore di un'unica condizione - vedi il docstring di
    average_relevant_elasticities - e mescolarla senza distinzione con
    medie vere su più condizioni nello stesso grafico è fuorviante;
    passa min_n_relevant=1 esplicitamente se vuoi comunque vederli).
    """
    subset = df_avg[
        (df_avg["system_name"] == system_name) &
        (df_avg["propulsor"] == propulsor) &
        (df_avg["output"] == output)
    ].copy()
    subset = subset[subset["n_relevant"] >= min_n_relevant]
    subset = subset.sort_values("mean_abs_elasticity")
    if top_n is not None:
        subset = subset.tail(top_n)

    if subset.empty:
        print(f"Nessun dato da plottare per system_name={system_name!r}, "
              f"propulsor={propulsor!r}, output={output!r}.")
        return

    labels = [f"{row.param_name}  (n={row.n_relevant}/{row.n_conditions_total})"
              for row in subset.itertuples()]
    values = subset["mean_abs_elasticity"].tolist()
    errs = subset["std_abs_elasticity"].fillna(0.0).tolist()

    if title is None:
        out_label = "Range massimo fattibile" if output == "max_range_nmi" else "Electricity Intensity"
        title = (f'Tornado Plot - Elasticità media assoluta (parametri rilevanti)\n'
                  f'{system_name} ({propulsor}) — {out_label}')

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, xerr=errs, color='#4c72b0', edgecolor='black', capsize=3)
    if reference_line is not None:
        plt.axvline(reference_line, color='black', linestyle='--', linewidth=1.2,
                    label=f'riferimento = {reference_line}')
        plt.legend(loc='lower right', fontsize=9)
    plt.xlabel('Media di |Indice di Elasticità|', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)

def plot_all_average_elasticities(df_avg: pd.DataFrame, top_n=None, min_n_relevant: int = 2, 
                                  reference_line: float = None):
    """Genera un plot_average_elasticity per ogni combinazione
    (system_name, propulsor, output) presente in df_avg, comodo per
    rivedere tutti i sistemi/propulsori in un colpo solo, invece di
    richiamare plot_average_elasticity singolarmente per ciascuno.
    Analoga a plot_all_condition_tornados, ma sui dati aggregati invece
    che sulle singole condizioni
    """
    combos = df_avg[["system_name", "propulsor", "output"]].drop_duplicates()
    for _, row in combos.iterrows():
        plot_average_elasticity(df_avg, system_name=row["system_name"], propulsor=row["propulsor"],
                                 output=row["output"], top_n=top_n, min_n_relevant=min_n_relevant, 
                                 reference_line=reference_line)

def average_relevant_elasticities_by_propulsor(df_elasticities: pd.DataFrame,
                                                 threshold: float) -> pd.DataFrame:
    """Wrapper di average_relevant_elasticities con group_cols=("propulsor",
    "output"): aggrega su tutti i sistemi e tutte le condizioni, invece
    che sistema per sistema."""
    return average_relevant_elasticities(df_elasticities, threshold=threshold,
                                          group_cols=("propulsor", "output"))


def plot_average_elasticity_by_propulsor(df_avg_propulsor: pd.DataFrame, propulsor: str,
                                          output: str = "intensity", top_n=None, min_n_relevant: int = 2,
                                          title: str = None, reference_line: float = None):
    """Tornado-style plot dell'elasticità media assoluta, aggregato su
    TUTTI i sistemi propulsivi che condividono lo stesso propulsor
    (fan o propeller), su tutte le condizioni operative."""
    subset = df_avg_propulsor[
        (df_avg_propulsor["propulsor"] == propulsor) &
        (df_avg_propulsor["output"] == output)
    ].copy()
    subset = subset[subset["n_relevant"] >= min_n_relevant]
    subset = subset.sort_values("mean_abs_elasticity")
    if top_n is not None:
        subset = subset.tail(top_n)

    if subset.empty:
        print(f"Nessun dato da plottare per propulsor={propulsor!r}, output={output!r}.")
        return

    labels = [f"{row.param_name}  (n={row.n_relevant}/{row.n_conditions_total})"
              for row in subset.itertuples()]
    values = subset["mean_abs_elasticity"].tolist()
    errs = subset["std_abs_elasticity"].fillna(0.0).tolist()

    if title is None:
        out_label = "Range massimo fattibile" if output == "max_range_nmi" else "Electricity Intensity"
        title = (f"Tornado Plot - Elasticità media assoluta (TUTTI i sistemi, parametri rilevanti)\n"
                  f"Propulsore: {propulsor} — {out_label}")

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, xerr=errs, color='#dd8452', edgecolor='black', capsize=3)
    if reference_line is not None:
        plt.axvline(reference_line, color='black', linestyle='--', linewidth=1.2,
                    label=f'riferimento = {reference_line}')
        plt.legend(loc='lower right', fontsize=9)
    plt.xlabel('Media di |Indice di Elasticità|', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)


def plot_all_average_elasticities_by_propulsor(df_avg_propulsor: pd.DataFrame, top_n=None,
                                                min_n_relevant: int = 2, reference_line: float = None):
    """Genera plot_average_elasticity_by_propulsor per ogni combinazione
    (propulsor, output): tipicamente due figure, fan e propeller."""
    combos = df_avg_propulsor[["propulsor", "output"]].drop_duplicates()
    for _, row in combos.iterrows():
        plot_average_elasticity_by_propulsor(df_avg_propulsor, propulsor=row["propulsor"],
                                              output=row["output"], top_n=top_n,
                                              min_n_relevant=min_n_relevant,
                                              reference_line=reference_line)