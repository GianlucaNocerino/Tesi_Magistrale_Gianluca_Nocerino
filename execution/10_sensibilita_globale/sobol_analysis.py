"""
Sensibilità globale (Sobol) della renewable electricity intensity
rispetto ai parametri tecnologici, su tutto il piano range-velocità.

Presuppone theta_acc.csv.

Uso:
    python execution/10_sensibilita_globale/sobol_analysis.py

Flusso:
  1. griglia (range, velocità) x 8 architetture = gli output
  2. pilota: scarta le colonne quasi sempre infattibili
  3. disegno di Saltelli sui SELECTED_FACTORS (+ gruppo "others"),
     valutazione del modello (in parallelo, con cache su disco)
  4. aggiunta della pseudo-architettura E_best = min sulle architetture
  5. indici S_1, S_T con bootstrap condiviso fra tutti gli output
  6. aggregazione (default: media pesata con Var(log E), cioè indice
     di Sobol generalizzato) e figure

Le valutazioni del modello (la parte costosa) restano in
sobol_cache/, un file per blocco del disegno (A, B, un AB per gruppo). Aggiungendo o togliendo fattori da SELECTED_FACTORS si
valutano solo i blocchi nuovi: tipicamente quello del fattore aggiunto
e quello del nuovo "others". Tutto si rivaluta solo se cambiano N, seed,
griglia o parametri del pilota. Per rifare SOLO le figure (colori,
font, titoli...) non serve nemmeno questo script:

    python execution/10_sensibilita_globale/sobol_figures.py

legge i CSV qui sotto e ridisegna tutto in pochi secondi.

Produce (in questa cartella):
  sobol_indices.csv         tidy: output x fattore (S1, ST, IC, n_valid, fattibile)
  sobol_outputs.csv         un output per riga: V, frazione_valida, fattibile, P_best
  sobol_aggregate.csv/.png  S_T aggregato su tutte le architetture
  influence_maps.csv        tutte le celle di tutte le mappe
  mappe/NN_<gruppo>.png     mappe del parametro più influente: complessiva,
                            fan / elica, per sistema, per architettura, E_best
  best_system_map.png       controllo: "most likely best" dagli stessi campioni
  best_system_map.csv
  sobol_convergence.csv/.png

Gli effetti di Shapley dei singoli parametri di calibrazione stanno in
shapley_calibration.py (stessa cartella), da lanciare dopo questo.
  sobol_cache/              valutazioni per blocco (FORCE_RECOMPUTE = True per rifarle)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from cnav.uncertainty import load_theta_acc
from cnav.sensitivity_global import (
    RESIDUAL_GROUP,
    FactorSpace,
    aggregate_indices,
    append_best_system,
    best_probability,
    best_system_map,
    define_groups,
    evaluate_outputs,
    grid_outputs,
    influence_map,
    nan_report,
    outputs_table,
    prescreen_outputs,
    residual_report,
    sobol_convergence,
    sobol_design,
    sobol_field,
    tail_report,
)

from sobol_figures import make_figures   # stesso folder dello script

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parent
MAP_DIR = FIGURE_DIR / "mappe"
THETA_ACC_PATH = FIGURE_DIR.parent / "06_incertezza_calibrazione" / "theta_acc.csv"
CACHE_DIR = FIGURE_DIR / "sobol_cache"            # valutazioni, un file per blocco
LEGACY_CACHE = FIGURE_DIR / "sobol_samples.npz"    # cache delle versioni precedenti

# I PARAMETRI DA ANALIZZARE, scelti a mano: quelli intrinsecamente
# incerti perché tecnologici. Ogni riga va motivata in tesi.
# Tutto ciò che non è qui (regressioni OEW, blocco di calibrazione, ...)
# resta campionato e finisce nel gruppo "others".
SELECTED_FACTORS = [
    # --- tecnologie di bordo ---
    "e_battery_Wh_per_kg",
    "fuel_cell_specific_power_kW_per_kg",
    "eta_fuel_cell",
    "gamma_tank",
    "eta_motor",
    "eta_p_propeller",
    "eta_p_fan",
    # --- filiera energetica (efficienze di produzione dei vettori) ---
    # sono tecnologiche anch'esse; toglierle sposta la loro varianza in
    # "others", che per e-SAF e H2 diventerebbe dominante
    "electricity",
    "liquid_hydrogen",          # include il fattore comune di elettrolisi (copula)
    "e_saf",                    # solo il residuo idiosincratico dell'e-SAF
    # --- non tecnologici, ma separabili dal "others" se interessa ---
    "calibration_block",      # i 6 parametri di calibrazione, in blocco
]

# Griglia del piano: stessa estensione della mappa "most likely best".
# Costo ~ righe x colonne: con 16 x 14 nodi x 8 architetture e N = 512
# sono ~6600 righe x (fino a) 1800 colonne prima del prescreen
RANGES_NMI = np.geomspace(10.0, 10000.0, 45)
SPEEDS_KT = np.arange(175.0, 501.0, 8.3)

N_BASE = 1024              # potenza di 2. 512 per le mappe; 1024-2048 se il tempo lo consente
N_BOOTSTRAP = 200          # intervalli per output nel CSV e baffi "bootstrap"; 0 per saltarlo
SEED = 1
TRANSFORM = "log"          # decompone Var(log E): vedi sobol.py e aggregation.py
NAN_TOL = 0.10             # sopra il 10% di NaN un output è "non fattibile"

PILOT_N = 64               # campione pilota del prescreen
PILOT_MAX_NAN = 0.50       # colonne con più NaN di così non si valutano

# pesi di aggregazione: "varianza" (default, indice generalizzato),
# "uniforme", "probabilita_migliore"
WEIGHTS = "varianza"
LEADER_INCLUDE_RESIDUAL = True

INCLUDE_BEST = True         # mappa aggiuntiva per E_best = min sulle architetture
BEST_CANDIDATES = None      # None = tutte le architetture; altrimenti lista di etichette


N_JOBS = max(1, (os.cpu_count() or 2) - 1)
FORCE_RECOMPUTE = False     # True per cancellare la cache e rivalutare tutto

# I fattori usati in un run lanciato con una versione PRECEDENTE di
# questo script (cache sobol_samples.npz). Serve solo a recuperare quelle
# valutazioni: se è la lista di default, non toccarla
PREVIOUS_SELECTED_FACTORS = [
    "e_battery_Wh_per_kg", "fuel_cell_specific_power_kW_per_kg", "eta_fuel_cell",
    "gamma_tank", "eta_motor", "eta_p_propeller", "eta_p_fan",
    "electricity", "liquid_hydrogen", "e_saf",
]


def _signature(space, all_outputs, include_theta_acc: bool = True) -> str:
    """Ciò che cambia le valutazioni di OGNI blocco: se cambia uno di
    questi (compreso il contenuto di theta_acc) la cache non vale più. SELECTED_FACTORS non c'è apposta: le
    matrici A e B dipendono solo da N, seed e numero di fattori, quindi
    cambiando i fattori selezionati si riusano tutti i blocchi già
    calcolati e si valutano solo quelli nuovi (vedi _block_keys)"""
    key = repr((N_BASE, SEED, space.names, PILOT_N, PILOT_MAX_NAN,
                [o.name for o in all_outputs]))
    if include_theta_acc and space.theta_acc is not None:
        # se si rifà la calibrazione cambiano le righe di theta_acc, e con
        # loro tutte le valutazioni: la cache non deve sopravvivere
        key += hashlib.sha1(np.ascontiguousarray(
            space.theta_acc.to_numpy(dtype=float)).tobytes()).hexdigest()
    return hashlib.sha1(key.encode()).hexdigest()


def _block_keys(design) -> list:
    """Un nome per ogni blocco del disegno, nell'ordine di design.U.

    Un blocco AB è identificato dalle COLONNE che prende da B, non dal
    nome del gruppo: AB per "eta_motor" è lo stesso blocco qualunque
    altro fattore sia selezionato, mentre il "others" cambia contenuto
    (e quindi chiave) quando gli si toglie un fattore"""
    keys = ["A", "B"]
    for cols in design.group_columns:
        keys.append("AB_" + "-".join(str(c) for c in sorted(cols)))
    return keys


# ---------------------------------------------------------------------
# Cache su disco
#
#   sobol_cache/meta.json          firma della configurazione + output tenuti
#   sobol_cache/<blocco>.npy       un blocco (N righe x output)
# ---------------------------------------------------------------------

def _atomic_save(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _write_meta(sig, columns):
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = CACHE_DIR / "meta.json.tmp"
    tmp.write_text(json.dumps({"signature": sig, "columns": list(columns)}))
    os.replace(tmp, CACHE_DIR / "meta.json")


def _read_meta(sig, sig_without_theta=None):
    """Gli output tenuti dal prescreen, se la cache è di questa
    configurazione; None altrimenti.

    sig_without_theta: la firma calcolata come nella versione precedente
    dello script (senza il contenuto di theta_acc). Una cache scritta da
    quella versione viene riconosciuta e aggiornata, non cancellata"""
    f = CACHE_DIR / "meta.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text())
    if sig_without_theta is not None and meta.get("signature") == sig_without_theta:
        _write_meta(sig, meta["columns"])
        print("\nCache della versione precedente riconosciuta e aggiornata")
        return meta["columns"]
    if meta.get("signature") != sig:
        print("\nCache di un'altra configurazione (N, seed, griglia, pilota o "
              "fattori dello spazio): si riparte da zero")
        for p in CACHE_DIR.glob("*.npy"):
            p.unlink()
        return None
    return meta["columns"]


def _migrate_legacy(sig, space, all_outputs):
    """Recupera le valutazioni salvate in sobol_samples.npz dalle
    versioni precedenti dello script, convertendole al formato a blocchi.
    Riconosce le tre firme usate finora; per le prime due serve sapere con
    quali fattori era stato lanciato (SELECTED_FACTORS attuale o
    PREVIOUS_SELECTED_FACTORS)"""
    if not LEGACY_CACHE.exists():
        return None
    z = np.load(LEGACY_CACHE, allow_pickle=True)   # file prodotto da te, sicuro
    if "signature" not in z.files or "columns" not in z.files:
        return None
    old_sig = str(z["signature"])
    columns = [str(c) for c in z["columns"]]

    # versione a blocchi in un unico .npz
    if (old_sig == _signature(space, all_outputs, include_theta_acc=False)
            and any(k.startswith("blk__") for k in z.files)):
        blocks = {k[len("blk__"):]: z[k] for k in z.files if k.startswith("blk__")}
    else:
        blocks = None
        all_names = [o.name for o in all_outputs]
        for sel in (SELECTED_FACTORS, PREVIOUS_SELECTED_FACTORS):
            if "Y" not in z.files:
                break
            sigs = {
                hashlib.sha1(repr((sel, N_BASE, SEED, space.names, columns)).encode()).hexdigest(),
                hashlib.sha1(repr((sel, N_BASE, SEED, space.names, PILOT_N, PILOT_MAX_NAN,
                                   all_names)).encode()).hexdigest(),
            }
            if old_sig in sigs:
                old = sobol_design(space, sel, N=N_BASE, seed=SEED)
                Yold = z["Y"].reshape(old.n_blocks, old.N, -1)
                blocks = dict(zip(_block_keys(old), Yold))
                break
    if blocks is None:
        print(f"\n{LEGACY_CACHE.name} trovato ma non corrisponde alla configurazione "
              "attuale (o a PREVIOUS_SELECTED_FACTORS): non lo uso")
        return None

    _write_meta(sig, columns)
    for k, v in blocks.items():
        _atomic_save(CACHE_DIR / f"{k}.npy", np.asarray(v, dtype=float))
    print(f"\nRecuperati {len(blocks)} blocchi da {LEGACY_CACHE.name} "
          f"(si può cancellare: ora la cache è in {CACHE_DIR.name}/)")
    return columns


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    MAP_DIR.mkdir(exist_ok=True)

    theta_acc = load_theta_acc(THETA_ACC_PATH)
    space = FactorSpace.default(theta_acc)

    # -----------------------------------------------------------------
    # Output e prescreen
    # -----------------------------------------------------------------
    all_outputs = grid_outputs(RANGES_NMI, SPEEDS_KT)
    ranges = sorted({o.range_nmi for o in all_outputs})
    speeds = sorted({o.speed_kt for o in all_outputs})

    print("=" * 78)
    print("GRIGLIA")
    print("=" * 78)
    print(f"{len(ranges)} range x {len(speeds)} velocità x 8 architetture = "
          f"{len(all_outputs)} output")

    sig = _signature(space, all_outputs)
    if FORCE_RECOMPUTE and CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*"):
            p.unlink()
        print("\nFORCE_RECOMPUTE: cache cancellata")
    kept = _read_meta(sig, _signature(space, all_outputs, include_theta_acc=False))
    if kept is None and not FORCE_RECOMPUTE:
        kept = _migrate_legacy(sig, space, all_outputs)

    if kept is not None:
        kept = set(kept)
        outputs = [o for o in all_outputs if o.name in kept]
        print(f"\nCache valida ({CACHE_DIR.name}/): prescreen saltato")
    else:
        t0 = time.time()
        outputs, dropped, _ = prescreen_outputs(space, all_outputs, n_pilot=PILOT_N,
                                                max_nan=PILOT_MAX_NAN, seed=SEED + 100,
                                                n_jobs=N_JOBS)
        print(f"Prescreen ({PILOT_N} campioni, {time.time() - t0:.0f}s): "
              f"{len(outputs)} output tenuti, {len(dropped)} scartati (>{PILOT_MAX_NAN:.0%} NaN)")
        _write_meta(sig, [o.name for o in outputs])
    meta = outputs_table(outputs)

    # -----------------------------------------------------------------
    # Disegno e valutazione
    # -----------------------------------------------------------------
    design = sobol_design(space, SELECTED_FACTORS, N=N_BASE, seed=SEED)
    print("\n" + "=" * 78)
    print("DISEGNO")
    print("=" * 78)
    print(design.cost_table().to_string(index=False))
    others = [nm for nm in space.names if nm not in SELECTED_FACTORS]
    print(f"\nAnalizzati singolarmente ({len(SELECTED_FACTORS)}): {', '.join(SELECTED_FACTORS)}")
    print(f"Nel gruppo '{RESIDUAL_GROUP}' ({len(others)}), campionati ma non separati: "
          f"{', '.join(others)}")

    # valutazione SOLO dei blocchi che non sono già in cache
    keys = _block_keys(design)
    labels = ["A", "B"] + list(design.group_names)
    U_blocks = design.U.reshape(design.n_blocks, design.N, -1)
    missing = [i for i, k in enumerate(keys) if not (CACHE_DIR / f"{k}.npy").exists()]
    if missing:
        print(f"\nDa valutare {len(missing)} blocchi su {len(keys)} "
              f"({', '.join(labels[i] for i in missing)}), {design.N} righe ciascuno "
              f"x {len(outputs)} output, {N_JOBS} processi")
        theta = space.theta_from_unit(np.vstack([U_blocks[i] for i in missing]))
        t0 = time.time()
        Ynew = evaluate_outputs(theta, outputs, n_jobs=N_JOBS, verbose=True).to_numpy()
        print(f"Tempo: {(time.time() - t0) / 60:.1f} min")
        for j, i in enumerate(missing):
            _atomic_save(CACHE_DIR / f"{keys[i]}.npy", Ynew[j * design.N:(j + 1) * design.N])
    else:
        print("\nTutti i blocchi del disegno sono già in cache: nessuna valutazione")
    Y = pd.DataFrame(np.vstack([np.load(CACHE_DIR / f"{k}.npy") for k in keys]),
                     columns=[o.name for o in outputs])

    print("\nNaN per architettura (punti con più del 10% di NaN):")
    print(nan_report(Y, meta).to_string())

    tails = tail_report(design, Y, nan_tol=NAN_TOL)
    print("\nCode (verdetti sul campione A+B):")
    print(tails["verdetto"].value_counts().to_string())

    # -----------------------------------------------------------------
    # E_best e P(best)
    # -----------------------------------------------------------------
    p_best = best_probability(design, Y, meta, candidates=BEST_CANDIDATES)
    if INCLUDE_BEST:
        Y, meta = append_best_system(Y, meta, candidates=BEST_CANDIDATES)

    # -----------------------------------------------------------------
    # Indici
    # -----------------------------------------------------------------
    t0 = time.time()
    fld = sobol_field(design, Y, meta.set_index("output"), n_boot=N_BOOTSTRAP,
                      seed=SEED, transform=TRANSFORM, nan_tol=NAN_TOL)
    print(f"\nIndici calcolati in {time.time() - t0:.0f}s")
    fld.table.to_csv(FIGURE_DIR / "sobol_indices.csv", index=False)
    fld.outputs.assign(P_best=p_best.reindex(fld.outputs["output"]).to_numpy()) \
        .to_csv(FIGURE_DIR / "sobol_outputs.csv", index=False)

    print("\n" + "=" * 78)
    print("QUOTA NON TECNOLOGICA: S_T[others] sugli output fattibili")
    print("=" * 78)
    print(residual_report(fld).round(3).to_string())

    # -----------------------------------------------------------------
    # Grafico aggregato
    # -----------------------------------------------------------------
    agg = aggregate_indices(fld, weights=WEIGHTS, p_best=p_best)
    agg = agg.assign(pesi=WEIGHTS, spread_lo=agg.attrs["spread"][0],
                     spread_hi=agg.attrs["spread"][1])
    agg.to_csv(FIGURE_DIR / "sobol_aggregate.csv", index=False)
    print("\n" + "=" * 78)
    print(f"S_T AGGREGATO (tutte le architetture, pesi = {WEIGHTS})")
    print("=" * 78)
    print(agg.round(3).to_string(index=False))

    # -----------------------------------------------------------------
    # Mappe (solo tabelle: le figure le fa sobol_figures.py)
    # -----------------------------------------------------------------
    groups = define_groups(fld.outputs, include_best=INCLUDE_BEST)
    maps = []
    for g in groups:
        mp = influence_map(fld, fld.mask(**g["criteri"]), ranges, speeds,
                           weights=WEIGHTS, p_best=p_best,
                           include_residual=LEADER_INCLUDE_RESIDUAL)
        maps.append(mp.assign(livello=g["livello"], gruppo=g["nome"], file=g["slug"],
                              pesi=WEIGHTS))
    pd.concat(maps, ignore_index=True).to_csv(FIGURE_DIR / "influence_maps.csv", index=False)

    best_system_map(p_best, meta, ranges, speeds) \
        .to_csv(FIGURE_DIR / "best_system_map.csv", index=False)

    # -----------------------------------------------------------------
    # Convergenza
    # -----------------------------------------------------------------
    conv = sobol_convergence(design, Y, meta, transform=TRANSFORM, nan_tol=NAN_TOL,
                             weights=WEIGHTS, p_best=p_best)
    conv.to_csv(FIGURE_DIR / "sobol_convergence.csv", index=False)

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------
    make_figures()   # stile e tipo di baffi: vedi sobol_figures.py


if __name__ == "__main__":
    main()
