"""
Propagazione dell'incertezza.

Dato il campione theta prodotto (cnav.uncertainty.distributions.assemble_theta), 
per ogni campione k si esegue il modello completo su tutta la griglia (R, V):

    y^(k) = f(R, V, theta^(k))

e per ogni punto della griglia si registra:
  - l'electricity intensity di ciascuna tecnologia, E_j^(k)(R, V),
    da cui si ricavano gli intervalli di confidenza richiesti
  - la tecnologia ottima T^(k)(R, V) = argmin_j E_j^(k)(R, V),
    da cui si costruisce la mappa probabilistica

Questo modulo si ferma qui: produce il "cubo" di risultati grezzi e lo
salva. Trasformarlo in probabilità, mappe e statistiche dei confini è
compito di cnav.uncertainty.technology_map, in modo da poter rifare
l'analisi (soglie diverse, aggregazioni diverse) senza rilanciare la
propagazione.

COSTO COMPUTAZIONALE
--------------------
Una valutazione del modello (una missione, un sistema propulsivo) costa
intorno a 0.7 ms. Un singolo campione su una griglia di n_R x n_V punti
costa quindi circa

    n_R * n_V * 2 propulsori * 4 sistemi * 0.7 ms

cioe' ~2.5 s per una griglia 28 x 16, e ~40 minuti per N = 1000 campioni
in sequenziale. Da qui due accorgimenti, entrambi accesi di default:

  - CHECKPOINT a blocchi su disco: se il run si interrompe si riparte da
    dove si era arrivati (resume=True)
  - PARALLELIZZAZIONE opzionale su più processi (n_workers): i campioni
    sono indipendenti, quindi lo speed-up e' quasi lineare nel numero di
    core fisici

Consiglio: prima un giro con N piccolo (50-100) per verificare il flusso 
e misurare il tempo per campione, poi il run "vero" da tesi lanciato offline.

"""
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..model.constants import TechAssumptions, WellToTankEfficiencies
from ..model.aircraft_sizer import AircraftSizer
from ..model.energy_intensity import compute_intensity
from ..model.mission import Mission
from ..model.propulsion_systems import build_default_systems
from ..model.well_to_tank import build_energy_carriers

__all__ = [
    "TECHNOLOGIES",
    "PROPULSORS",
    "FlightGrid",
    "technology_labels",
    "theta_row_to_tech_wtt",
    "evaluate_sample",
    "PropagationResult",
    "run_propagation",
]

TECHNOLOGIES = ("Battery-electric", "Hydrogen fuel cell",
                "Hydrogen combustion", "e-SAF combustion")
PROPULSORS = ("fan", "propeller")


def technology_labels() -> list:
    """Le 8 combinazioni tecnologia x propulsore, nell'ordine usato in
    tutti gli array di questo modulo.

    La distinzione fan/elica è mantenuta nel risultato grezzo perchè è
    quella della mappa deterministica dell'articolo di riferimento;
    l'aggregazione alle 4 tecnologie "pure" è un'operazione a valle
    (technology_map.aggregate_to_technologies) e resta reversibile
    """
    return [f"{tech} ({prop})" for tech in TECHNOLOGIES for prop in PROPULSORS]


@dataclass(frozen=True)
class FlightGrid:
    """La griglia di condizioni operative (R, V) su cui si valuta il modello.

    Il default è spaziato logaritmicamente in range (i confini fra
    tecnologie si addensano ai raggi corti) e linearmente in velocità.

    Attenzione al costo: il tempo di calcolo cresce come il prodotto
    n_ranges * n_speeds. Raddoppiare entrambi quadruplica il run. Per il
    risultato da tesi conviene infittire la griglia solo nella zona dei
    confini fra tecnologie, dove serve risoluzione, invece che ovunque
    """
    ranges_nmi: np.ndarray
    speeds_kt: np.ndarray

    @classmethod
    def default(cls, n_ranges: int = 45, n_speeds: int = 40,
                range_bounds: tuple = (10.0, 10_000.0),
                speed_bounds: tuple = (180.0, 500.0)) -> "FlightGrid":
        """Stessi estremi di examples/plot_best_system_map*.py (10-10000
        nmi, 180-500 kt), con meno punti: la mappa probabilistica costa N
        volte quella deterministica. Estremi identici servono a poter
        sovrapporre le due mappe senza reinterpolare"""
        return cls(
            ranges_nmi=np.geomspace(range_bounds[0], range_bounds[1], n_ranges),
            speeds_kt=np.linspace(speed_bounds[0], speed_bounds[1], n_speeds),
        )

    @property
    def shape(self) -> tuple:
        """(n_speeds, n_ranges): l'ordine degli assi in tutti gli array,
        scelto per essere direttamente compatibile con pcolormesh(R, V, Z)"""
        return len(self.speeds_kt), len(self.ranges_nmi)

    def n_model_evaluations(self) -> int:
        return len(self.speeds_kt) * len(self.ranges_nmi) * len(PROPULSORS) * len(TECHNOLOGIES)


def theta_row_to_tech_wtt(row) -> tuple:
    """Da una riga della matrice theta a (TechAssumptions, WellToTankEfficiencies).

    Ogni nome di colonna deve corrispondere a un campo dell'una o
    dell'altra dataclass; i campi non presenti in theta restano al loro
    valore nominale. Sarebbe la stessa logica di
    cnav.calibration.deterministic_calibration.theta_to_tech_wtt, qui
    riscritta per accettare un dict/Series invece di un vettore piatto
    più la lista dei nomi
    """
    base_tech, base_wtt = TechAssumptions(), WellToTankEfficiencies()
    tech_kwargs, wtt_kwargs = {}, {}
    for name, value in dict(row).items():
        if hasattr(base_tech, name):
            tech_kwargs[name] = float(value)
        elif hasattr(base_wtt, name):
            wtt_kwargs[name] = float(value)
        else:
            raise ValueError(
                f"Parametro sconosciuto: {name!r}. Deve essere un campo di "
                "TechAssumptions o di WellToTankEfficiencies")
    return base_tech.with_changes(**tech_kwargs), base_wtt.with_changes(**wtt_kwargs)


def evaluate_sample(row, grid: FlightGrid) -> np.ndarray:
    """Valuta un singolo campione theta su tutta la griglia.

    Ritorna un array (8, n_speeds, n_ranges) float32 con l'electricity
    intensity [MJ/(pax*nmi)] di ogni combinazione tecnologia-propulsore.

    NaN significa "configurazione non fattibile per questo theta in
    questo punto" (tipicamente la batteria oltre il suo range massimo:
    il ciclo di dimensionamento non converge e compute_intensity
    restituisce NaN). I NaN NON vanno sostituiti con valori grandi: sono
    l'informazione che serve per ricostruire il limite di fattibilita'
    probabilistico, e argmin li ignora comunque
    """
    tech, wtt = theta_row_to_tech_wtt(row)
    sizer = AircraftSizer(tech)
    systems = build_default_systems(build_energy_carriers(wtt))

    n_speeds, n_ranges = grid.shape
    out = np.full((len(TECHNOLOGIES) * len(PROPULSORS), n_speeds, n_ranges),
                  np.nan, dtype=np.float32)

    for i, speed_kt in enumerate(grid.speeds_kt):
        for j, range_nmi in enumerate(grid.ranges_nmi):
            for p_idx, propulsor in enumerate(PROPULSORS):
                mission = Mission(range_nmi=float(range_nmi),
                                  cruise_speed_kt=float(speed_kt),
                                  propulsor=propulsor)
                for s_idx, system in enumerate(systems):
                    label_idx = s_idx * len(PROPULSORS) + p_idx
                    try:
                        result = compute_intensity(mission, system, tech, sizer)
                        out[label_idx, i, j] = result.intensity_MJ_per_pax_nmi
                    except Exception:
                        # un theta patologico non deve far cadere l'intero
                        # run: resta NaN, cioe' "non fattibile", e la cosa
                        # e' visibile nella frazione di NaN riportata a fine run
                        out[label_idx, i, j] = np.nan
    return out


def _best_index(intensities: np.ndarray) -> np.ndarray:
    """argmin sulle tecnologie ignorando i NaN (eq. 24).

    Ritorna int8 con -1 dove NESSUNA configurazione è fattibile
    """
    all_nan = np.all(np.isnan(intensities), axis=0)
    safe = np.where(np.isnan(intensities), np.inf, intensities)
    best = np.argmin(safe, axis=0).astype(np.int8)
    best[all_nan] = -1
    return best


# --- worker per la parallelizzazione ---------------------------------
_WORKER_GRID: Optional[FlightGrid] = None


def _worker_init(grid: FlightGrid):
    global _WORKER_GRID
    _WORKER_GRID = grid


def _worker_eval(args):
    sample_idx, row = args
    intensities = evaluate_sample(row, _WORKER_GRID)
    return sample_idx, intensities


@dataclass
class PropagationResult:
    """Il "cubo" grezzo della propagazione.

    intensities: (N, 8, n_speeds, n_ranges) float32, NaN dove non fattibile.
    best_index:  (N, n_speeds, n_ranges) int8, indice in labels, -1 se
                 nessuna configurazione è fattibile.
    theta:       la matrice dei campioni effettivamente valutati
    """
    intensities: np.ndarray
    best_index: np.ndarray
    theta: pd.DataFrame
    grid: FlightGrid
    labels: list

    @property
    def n_samples(self) -> int:
        return self.best_index.shape[0]

    @property
    def infeasible_fraction(self) -> float:
        """Frazione di celle (campione, punto di griglia) in cui nessuna
        tecnologia è fattibile. Se non e' ~0 c'è qualcosa da spiegare:
        di norma l'e-SAF è fattibile ovunque nella griglia"""
        return float(np.mean(self.best_index < 0))

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            intensities=self.intensities,
            best_index=self.best_index,
            ranges_nmi=self.grid.ranges_nmi,
            speeds_kt=self.grid.speeds_kt,
            labels=np.array(self.labels, dtype=object),
            theta_values=self.theta.to_numpy(),
            theta_columns=np.array(list(self.theta.columns), dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "PropagationResult":
        data = np.load(path, allow_pickle=True)
        theta = pd.DataFrame(data["theta_values"],
                             columns=list(data["theta_columns"]))
        return cls(
            intensities=data["intensities"],
            best_index=data["best_index"],
            theta=theta,
            grid=FlightGrid(ranges_nmi=data["ranges_nmi"], speeds_kt=data["speeds_kt"]),
            labels=list(data["labels"]),
        )


def _chunk_path(checkpoint_dir: Path, start: int) -> Path:
    return checkpoint_dir / f"chunk_{start:07d}.npz"


def _load_checkpoints(checkpoint_dir: Path, n_samples: int, grid: FlightGrid):
    """Rilegge i blocchi già calcolati. Ritorna (intensities, done_mask)"""
    n_speeds, n_ranges = grid.shape
    n_labels = len(TECHNOLOGIES) * len(PROPULSORS)
    intensities = np.full((n_samples, n_labels, n_speeds, n_ranges), np.nan, dtype=np.float32)
    done = np.zeros(n_samples, dtype=bool)

    for chunk_file in sorted(checkpoint_dir.glob("chunk_*.npz")):
        data = np.load(chunk_file)
        idx = data["sample_idx"]
        valid = idx < n_samples
        intensities[idx[valid]] = data["intensities"][valid]
        done[idx[valid]] = True
    return intensities, done


def run_propagation(theta: pd.DataFrame,
                    grid: Optional[FlightGrid] = None,
                    checkpoint_dir: Optional[str] = None,
                    chunk_size: int = 25,
                    n_workers: int = 1,
                    resume: bool = True,
                    verbose: bool = True) -> PropagationResult:
    """Esegue la propagazione su tutti i campioni di theta.

    theta: DataFrame (N, n_theta) da assemble_theta. Una riga = un
    campione, una colonna = un parametro (nome di un campo di
    TechAssumptions o WellToTankEfficiencies).

    grid: se None, FlightGrid.default().

    checkpoint_dir: cartella in cui salvare blocchi di chunk_size
    campioni per volta. Con resume=True, i blocchi gia' presenti NON
    vengono ricalcolati: è così che si riprende un run interrotto.
    Attenzione: la corrispondenza fra blocchi salvati e campioni si basa
    sull'INDICE di riga, quindi riprendere un run ha senso solo se theta
    è identica (stesso seed, stesso N, stessi specs). Cambiando theta
    va usata una cartella nuova, altrimenti si mescolano risultati di
    campioni diversi.

    n_workers: numero di processi. 1 = sequenziale (piu' semplice da
    interrompere e da profilare). Con n_workers > 1 lo script chiamante
    DEVE proteggere il codice con `if __name__ == "__main__":`,
    altrimenti i processi figli rieseguono lo script
    """
    grid = grid or FlightGrid.default()
    labels = technology_labels()
    n_samples = len(theta)

    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir_path is not None:
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir_path is not None and resume:
        intensities, done = _load_checkpoints(checkpoint_dir_path, n_samples, grid)
        if verbose and done.any():
            print(f"  Checkpoint trovati: {int(done.sum())}/{n_samples} campioni già valutati")
    else:
        n_speeds, n_ranges = grid.shape
        intensities = np.full((n_samples, len(labels), n_speeds, n_ranges),
                              np.nan, dtype=np.float32)
        done = np.zeros(n_samples, dtype=bool)

    todo = np.flatnonzero(~done)
    if verbose:
        print(f"Propagazione: {len(todo)} campioni da valutare "
              f"({grid.n_model_evaluations()} valutazioni del modello ciascuno, "
              f"{n_workers} worker)")

    t0 = time.time()
    n_done_session = 0

    for chunk_start in range(0, len(todo), chunk_size):
        chunk_idx = todo[chunk_start:chunk_start + chunk_size]
        payload = [(int(i), theta.iloc[i].to_dict()) for i in chunk_idx]

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_worker_init,
                                     initargs=(grid,)) as pool:
                results = list(pool.map(_worker_eval, payload))
        else:
            results = [(idx, evaluate_sample(row, grid)) for idx, row in payload]

        chunk_indices = np.array([r[0] for r in results], dtype=int)
        chunk_values = np.stack([r[1] for r in results])
        intensities[chunk_indices] = chunk_values
        done[chunk_indices] = True
        n_done_session += len(results)

        if checkpoint_dir_path is not None:
            np.savez_compressed(_chunk_path(checkpoint_dir_path, int(chunk_indices[0])),
                                sample_idx=chunk_indices, intensities=chunk_values)

        if verbose:
            elapsed = time.time() - t0
            rate = elapsed / n_done_session
            eta_min = rate * (len(todo) - n_done_session) / 60.0
            print(f"  [{int(done.sum())}/{n_samples}] "
                  f"{elapsed:.0f}s trascorsi, ~{rate:.2f}s/campione, ETA ~{eta_min:.0f} min")

    best_index = np.stack([_best_index(intensities[k]) for k in range(n_samples)])

    result = PropagationResult(intensities=intensities, best_index=best_index,
                               theta=theta, grid=grid, labels=labels)

    if verbose:
        print(f"\nCompletata in {time.time() - t0:.0f}s "
              f"(questa sessione: {n_done_session} campioni)")
        if result.infeasible_fraction > 0:
            print(f"  celle senza NESSUNA tecnologia fattibile: "
                  f"{100 * result.infeasible_fraction:.2f}%")
    return result
