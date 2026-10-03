"""
Gli output scalari Y su cui si calcolano gli indici.

La grandezza è una sola: la renewable electricity intensity
[MJ/(pax*nmi)], la stessa del modello deterministico.

Perchè tante colonne
--------------------
Sobol vuole uno scalare per campione, mentre l'intensity è una
superficie E(R, V) per ciascuna architettura. La superficie si campiona
su una GRIGLIA regolare del piano range-velocità (log-spaziata in
range, lineare in velocità, come le mappe del "most likely best"), e
ogni colonna di Y è l'intensity di un'architettura in un nodo:

    n_colonne = n_architetture x n_range x n_velocità

Architettura = (sistema propulsivo, propulsore), quindi 4 x 2 = 8. Le
colonne condividono il campione theta: tech, wtt, sizer e sistemi si
costruiscono una volta per riga e si riusano su tutte le colonne.

Il costo e come contenerlo
--------------------------
Le righe sono N*(g+2) (g = fattori selezionati + others), le colonne
qualche centinaio. Tre accorgimenti:

  1. prescreen_outputs: un campione pilota piccolo scarta in anticipo
     le colonne quasi sempre infattibili (la batteria a 3000 nmi). Sono
     le più care, perché il sizing non converge e itera fino al limite
  2. evaluate_outputs(n_jobs=...): le righe sono indipendenti, si
     dividono fra processi
  3. la cache su disco nello script: le figure si rifanno senza
     rivalutare il modello

I NaN
-----
Una configurazione infattibile dà NaN, e i NaN non mancano a caso:
mancano dove i parametri sono sfavorevoli. Non si riempiono: le
colonne con più del nan_tol di NaN vengono marcate non fattibili
(sobol.py) e nelle mappe compaiono come zona tratteggiata.

MISSION_LIBRARY e REPRESENTATIVE_MISSIONS restano per compatibilità
(e per confrontare con l'analisi locale), ma l'analisi globale ora
lavora sulla griglia.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..model.aircraft_sizer import AircraftSizer
from ..model.energy_intensity import compute_intensity
from ..model.mission import Mission
from ..model.propulsion_systems import build_default_systems
from ..model.well_to_tank import build_energy_carriers
from ..uncertainty.propagation import theta_row_to_tech_wtt

__all__ = [
    "SYSTEMS",
    "PROPULSORS",
    "BEST_SYSTEM",
    "IntensityAt",
    "MISSION_LIBRARY",
    "REPRESENTATIVE_MISSIONS",
    "architecture_label",
    "grid_outputs",
    "outputs_from_missions",
    "default_outputs",
    "outputs_table",
    "evaluate_outputs",
    "prescreen_outputs",
    "append_best_system",
    "nan_report",
]

# i sistemi propulsivi del modello (i nomi di build_default_systems)
SYSTEMS = [
    "Battery-electric",
    "Hydrogen fuel cell",
    "Hydrogen combustion",
    "e-SAF combustion",
]
PROPULSORS = ["propeller", "fan"]

# nome della pseudo-architettura "il migliore fra i candidati" (vedi
# append_best_system)
BEST_SYSTEM = "Best system"


def architecture_label(system_name: str, propulsor: str) -> str:
    return f"{system_name} ({propulsor})"


# ---------------------------------------------------------------------
# Libreria delle missioni dell'analisi locale (compatibilità)
# ---------------------------------------------------------------------
MISSION_LIBRARY = {
    "battery_very_short_low_speed": Mission(range_nmi=10.0,   cruise_speed_kt=200.0, propulsor="propeller"),
    "battery_very_short":           Mission(range_nmi=10.0,   cruise_speed_kt=250.0, propulsor="propeller"),
    "battery_short":                Mission(range_nmi=20.0,   cruise_speed_kt=250.0, propulsor="propeller"),
    "short_low_speed_propeller":    Mission(range_nmi=100.0,  cruise_speed_kt=200.0, propulsor="propeller"),
    "short_high_speed_propeller":   Mission(range_nmi=100.0,  cruise_speed_kt=300.0, propulsor="propeller"),
    "medium_propeller":             Mission(range_nmi=1500.0, cruise_speed_kt=250.0, propulsor="propeller"),
    "long_propeller":               Mission(range_nmi=6000.0, cruise_speed_kt=250.0, propulsor="propeller"),
    "battery_very_short_fan":       Mission(range_nmi=5.0,    cruise_speed_kt=300.0, propulsor="fan"),
    "short_low_speed_fan":          Mission(range_nmi=100.0,  cruise_speed_kt=350.0, propulsor="fan"),
    "short_high_speed_fan":         Mission(range_nmi=100.0,  cruise_speed_kt=450.0, propulsor="fan"),
    "medium_fan":                   Mission(range_nmi=1500.0, cruise_speed_kt=450.0, propulsor="fan"),
    "long_fan":                     Mission(range_nmi=6000.0, cruise_speed_kt=450.0, propulsor="fan"),
}

REPRESENTATIVE_MISSIONS = {
    "Battery-electric": ["battery_very_short_low_speed", "battery_very_short",
                         "battery_short", "battery_very_short_fan"],
    "Hydrogen fuel cell": ["short_low_speed_propeller", "short_high_speed_propeller",
                           "medium_propeller", "long_propeller",
                           "short_low_speed_fan", "short_high_speed_fan"],
    "Hydrogen combustion": ["short_low_speed_fan", "short_high_speed_fan",
                            "medium_fan", "long_fan", "medium_propeller"],
    "e-SAF combustion": ["short_low_speed_fan", "medium_fan", "long_fan",
                         "medium_propeller"],
}


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def _intensity(system_name, mission, systems, tech, sizer) -> float:
    """Intensity di un sistema, NaN se non fattibile o se il modello
    solleva (un theta patologico non deve far cadere il run)"""
    system = next(s for s in systems if s.name == system_name)
    try:
        return float(compute_intensity(mission, system, tech, sizer).intensity_MJ_per_pax_nmi)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class IntensityAt:
    """Renewable electricity intensity [MJ/(pax*nmi)] di un'architettura
    (sistema + propulsore) in un punto (range, velocità)"""
    system_name: str
    range_nmi: float
    speed_kt: float
    propulsor: str
    mission_label: str = ""

    @property
    def name(self) -> str:
        return (f"E[{self.system_name}]@{self.range_nmi:g}nmi_"
                f"{self.speed_kt:g}kt_{self.propulsor}")

    @property
    def architecture(self) -> str:
        return architecture_label(self.system_name, self.propulsor)

    @property
    def mission(self) -> Mission:
        return Mission(range_nmi=self.range_nmi, cruise_speed_kt=self.speed_kt,
                       propulsor=self.propulsor)

    def evaluate(self, tech, wtt, systems, sizer) -> float:
        return _intensity(self.system_name, self.mission, systems, tech, sizer)


def _round_sig(x, sig: int = 3):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
    return np.round(x / 10 ** (mag - sig + 1)) * 10 ** (mag - sig + 1)


def grid_outputs(ranges_nmi, speeds_kt, systems: Optional[list] = None,
                 propulsors: Optional[list] = None) -> list:
    """Un IntensityAt per ogni (sistema, propulsore, range, velocità).

    I range vengono arrotondati a 3 cifre significative (nomi di colonna
    leggibili, nessun effetto pratico sulla mappa)"""
    systems = systems or SYSTEMS
    propulsors = propulsors or PROPULSORS
    ranges_nmi = _round_sig(ranges_nmi)
    out = []
    for s in systems:
        for p in propulsors:
            for v in speeds_kt:
                for r in ranges_nmi:
                    out.append(IntensityAt(system_name=s, range_nmi=float(r),
                                           speed_kt=float(v), propulsor=p,
                                           mission_label="grid"))
    return out


def outputs_from_missions(assignment: Optional[dict] = None,
                          library: Optional[dict] = None) -> list:
    """IntensityAt da una tabella {sistema: [etichette di missione]}"""
    assignment = assignment or REPRESENTATIVE_MISSIONS
    library = library or MISSION_LIBRARY
    outputs = []
    for system_name, labels in assignment.items():
        for label in labels:
            if label not in library:
                raise ValueError(f"Missione sconosciuta: {label!r}")
            m = library[label]
            outputs.append(IntensityAt(system_name=system_name, range_nmi=m.range_nmi,
                                       speed_kt=m.cruise_speed_kt, propulsor=m.propulsor,
                                       mission_label=label))
    return outputs


def default_outputs() -> list:
    """Le missioni rappresentative dell'analisi locale (compatibilità)"""
    return outputs_from_missions()


def outputs_table(outputs: list) -> pd.DataFrame:
    """I metadati degli output: sono le colonne su cui si raggruppa per
    aggregare (sistema, propulsore, architettura, range, velocità)"""
    return pd.DataFrame([{"output": o.name, "sistema": o.system_name,
                          "propulsore": o.propulsor, "architettura": o.architecture,
                          "range_nmi": o.range_nmi, "velocita_kt": o.speed_kt,
                          "missione": o.mission_label}
                         for o in outputs])


# ---------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------

def _evaluate_serial(theta: pd.DataFrame, outputs: list, verbose: bool = False,
                     progress_every: int = 200) -> np.ndarray:
    values = np.full((len(theta), len(outputs)), np.nan)
    records = theta.to_dict("records")
    for i, row in enumerate(records):
        tech, wtt = theta_row_to_tech_wtt(row)
        sizer = AircraftSizer(tech)
        systems = build_default_systems(build_energy_carriers(wtt))
        for j, out in enumerate(outputs):
            try:
                values[i, j] = out.evaluate(tech, wtt, systems, sizer)
            except Exception:
                values[i, j] = np.nan
        if verbose and (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(theta)} righe")
    return values


def _evaluate_chunk(args) -> np.ndarray:
    theta, outputs = args
    return _evaluate_serial(theta, outputs)


def evaluate_outputs(theta: pd.DataFrame, outputs: list, n_jobs: int = 1,
                     chunk_size: int = 100, verbose: bool = False,
                     progress_every: int = 500) -> pd.DataFrame:
    """Valuta tutti gli output su tutte le righe di theta.

    n_jobs > 1 divide le righe fra processi (blocchi di chunk_size
    righe). Su Windows funziona solo se lo script chiamante ha la guardia
    if __name__ == "__main__" (lo script di analisi ce l'ha); in un
    notebook conviene n_jobs=1
    """
    names = [o.name for o in outputs]
    if n_jobs is None or n_jobs <= 1 or len(theta) <= chunk_size:
        values = _evaluate_serial(theta, outputs, verbose=verbose,
                                  progress_every=progress_every)
        return pd.DataFrame(values, columns=names)

    chunks = [theta.iloc[i:i + chunk_size] for i in range(0, len(theta), chunk_size)]
    parts, done = [], 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for part in ex.map(_evaluate_chunk, [(c, outputs) for c in chunks]):
            parts.append(part)
            done += len(part)
            if verbose and (done // chunk_size) % max(1, progress_every // chunk_size) == 0:
                print(f"    {done}/{len(theta)} righe")
    return pd.DataFrame(np.vstack(parts), columns=names)


def prescreen_outputs(space, outputs: list, n_pilot: int = 64, max_nan: float = 0.5,
                      seed: int = 123, n_jobs: int = 1) -> tuple:
    """Campione pilota dalle PDF: scarta le colonne con più di max_nan
    di NaN. Ritorna (tenuti, scartati, frazione_nan_pilota).

    La soglia è larga apposta (0.5): il pilota deve solo evitare di
    pagare colonne sicuramente infattibili, la decisione vera la prende
    nan_tol sul campione completo
    """
    from scipy.stats import qmc
    U = qmc.Sobol(d=space.k, scramble=True, seed=seed).random(n_pilot)
    Y = evaluate_outputs(space.theta_from_unit(U), outputs, n_jobs=n_jobs,
                         chunk_size=max(8, n_pilot // max(1, n_jobs)))
    frac = Y.isna().mean()
    keep = [o for o in outputs if frac[o.name] <= max_nan]
    drop = [o for o in outputs if frac[o.name] > max_nan]
    return keep, drop, frac


def append_best_system(Y: pd.DataFrame, meta: pd.DataFrame,
                       candidates: Optional[list] = None) -> tuple:
    """Aggiunge, per ogni nodo (range, velocità), la colonna
    E_best = min sulle architetture candidate dell'intensity.

    È l'intensity dell'architettura migliore IN QUEL CAMPIONE: un output
    scalare vero, i cui indici di Sobol dicono quale parametro governa
    l'incertezza della soluzione migliore, tenendo conto anche del fatto
    che al variare dei parametri il migliore può cambiare. Le
    architetture infattibili in un campione (NaN) non sono candidate in
    quel campione. Costa zero: si ricava dalle colonne già calcolate.

    candidates: lista di etichette di architettura; None = tutte
    """
    m = meta[meta["sistema"] != BEST_SYSTEM]
    if candidates is not None:
        m = m[m["architettura"].isin(candidates)]
    new_cols, new_meta = {}, []
    for (r, v), cell in m.groupby(["range_nmi", "velocita_kt"], sort=False):
        vals = Y[cell["output"].tolist()].to_numpy()
        with np.errstate(invalid="ignore"):
            best = np.where(np.isfinite(vals).any(axis=1),
                            np.nanmin(np.where(np.isfinite(vals), vals, np.inf), axis=1),
                            np.nan)
        name = f"E[{BEST_SYSTEM}]@{r:g}nmi_{v:g}kt"
        new_cols[name] = best
        new_meta.append({"output": name, "sistema": BEST_SYSTEM, "propulsore": "-",
                         "architettura": BEST_SYSTEM, "range_nmi": r,
                         "velocita_kt": v, "missione": "grid"})
    Y2 = pd.concat([Y, pd.DataFrame(new_cols, index=Y.index)], axis=1)
    meta2 = pd.concat([meta, pd.DataFrame(new_meta)], ignore_index=True)
    return Y2, meta2


def nan_report(Y: pd.DataFrame, meta: Optional[pd.DataFrame] = None,
               by: str = "architettura") -> pd.DataFrame:
    """Frazione di NaN. Con meta, riassunta per architettura (con
    centinaia di colonne la tabella per colonna non si legge)"""
    frac = Y.isna().mean()
    if meta is None:
        return frac.rename("frazione_nan").to_frame().sort_values("frazione_nan")
    df = meta.set_index("output").reindex(frac.index)
    df["frazione_nan"] = frac
    return (df.groupby(by)["frazione_nan"]
            .agg(punti="size", mediana="median",
                 sopra_10pc=lambda s: int((s > 0.10).sum()))
            .sort_values("mediana"))
