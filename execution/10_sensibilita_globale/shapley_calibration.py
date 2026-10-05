"""
Effetti di Shapley dei singoli parametri di calibrazione, e figure della
Sobol con i 6 parametri al posto del blocco di calibrazione.

Tutto in un file: lo stimatore, l'aggregazione, le figure e il lancio.

Uso (DOPO sobol_analysis.py, con "calibration_block" fra i
SELECTED_FACTORS):
    python execution/10_sensibilita_globale/shapley_calibration.py

Non valuta il modello: legge i blocchi A e B salvati da sobol_analysis.py
in sobol_cache/, gli indici in sobol_indices.csv, e ne usa la stessa
configurazione (griglia, N, seed, trasformazione, NAN_TOL, gruppo
"others" fra i candidati delle mappe), importandola da lì. Pochi
secondi: si rilancia anche solo per ritoccare le figure.

Produce, nella sottocartella shapley/:
  shapley_indices.csv          per output e parametro: effetto di Shapley e
                               indice "S_T equivalente" (vedi sotto)
  shapley_aggregate.csv/.png   effetti di Shapley dei 6 parametri, aggregati
  calibration_correlation.csv/.png   correlazioni fra i parametri accettati
  sobol_aggregate.csv/.png     il grafico a barre della Sobol con i 6 parametri
                               al posto di calibration_block
  influence_maps.csv, mappe/   le mappe della Sobol, idem


1. PERCHÉ SHAPLEY
-----------------
I 6 parametri di calibrazione sono le righe accettate di theta_acc:
correlati e senza distribuzioni separate. Gli indici di Sobol dei singoli
non sono definiti con input correlati (per questo nella Sobol entrano
come un unico fattore di gruppo). Gli effetti di Shapley (Owen 2014;
Song, Nelson, Staum 2016) sono la generalizzazione per input correlati:
ripartiscono la varianza in modo univoco, dividendo equamente fra i
parametri la parte che condividono.

2. LA DEFINIZIONE USATA
-----------------------
Giocatori: i 6 parametri di calibrazione. Per ogni sottoinsieme S

    val(S) = Var( E[Y | c_S] ) / Var(Y),    Y = log(intensity)

(i parametri di letteratura restano incerti e vengono mediati: sono
indipendenti dal blocco per costruzione). L'effetto di c_j è

    Sh_j = sum_{S ⊆ P\\{j}}  |S|! (p-|S|-1)! / p!  [ val(S ∪ {j}) - val(S) ]

e per costruzione sum_j Sh_j = val(tutti) = S_1[calibration_block].

3. CONFRONTABILITÀ CON GLI INDICI TOTALI DI SOBOL
-------------------------------------------------
Con input indipendenti vale S_i <= Sh_i <= S_Ti (Owen 2014): l'effetto
di Shapley sta fra primo ordine e totale, perché divide ogni interazione
in parti uguali fra i parametri coinvolti invece di attribuirla per
intero a ciascuno. Quando le interazioni sono piccole i tre coincidono.

Qui gli Sh_j sommano a S_1 del blocco, mentre nelle figure della Sobol
il blocco compare con il suo S_T. Per metterli sullo stesso piano dei
parametri tecnologici, l'S_T del blocco viene ripartito fra i 6 in
proporzione ai loro effetti di Shapley:

    S_T,j^eq = S_T[calibration_block] * Sh_j / sum_k Sh_k

Così le 6 barre sommano esattamente a quella del blocco che sostituiscono
e ogni parametro tecnologico resta com'era. L'unica ipotesi è che la
parte di interazione del blocco (S_T - S_1, piccola: controllarla nel
riepilogo a video) si ripartisca fra i 6 come la parte diretta. Gli Sh_j
leggermente negativi (rumore di stima attorno a zero) valgono zero nella
ripartizione.

4. LO STIMATORE (dai dati, nessuna valutazione nuova)
-----------------------------------------------------
Campione: le 2N righe dei blocchi A e B, estrazioni indipendenti dalla
distribuzione congiunta di tutti i parametri. Per ogni sottoinsieme S

    E[ Var(Y | c_S) ] ≈ 1/(2n) sum_i ( y_i - y_{nn_S(i)} )^2

dove nn_S(i) è il campione più vicino a i guardando SOLO le coordinate in
S, portate ai ranghi (Devroye et al. 2018; per gli effetti di Shapley
Broto, Bachoc, Depecker 2020). Per i sottoinsiemi stretti il vicino deve
avere una riga di theta_acc DIVERSA: una riga ripetuta è identica in
tutte le coordinate e farebbe valere val(S) = val(tutti) per ogni S.

Verifica: su un modello lineare con 6 parametri gaussiani fortemente
correlati (effetti di Shapley noti in forma chiusa), con 2N = 1024 le
stime differiscono dai valori esatti di 0.01-0.03, con ordinamento
corretto. Sui dati veri: sum_j Sh_j contro S_1[calibration_block] della
Sobol, stimato con un metodo indipendente (stampato a video).

5. AGGREGAZIONE
---------------
Come per la Sobol: media pesata con V_k = Var(log E_k) sugli output
fattibili; nelle mappe, cella per cella fra le architetture del gruppo.

Bibliografia:
  Owen (2014), SIAM/ASA J. Uncertainty Quantification 2, 245-251
  Song, Nelson, Staum (2016), SIAM/ASA JUQ 4, 1060-1083
  Iooss, Prieur (2019), Int. J. Uncertainty Quantification 9, 493-514
  Broto, Bachoc, Depecker (2020), SIAM/ASA JUQ 8, 693-716
  Devroye, Györfi, Lugosi, Walk (2018), Electronic J. Statistics 12
"""
import json
import sys
import time
from itertools import combinations
from math import factorial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sobol_analysis as SA        # stessa configurazione e stessa cache della Sobol
from cnav.uncertainty import load_theta_acc
from cnav.sensitivity_global import (
    FACTOR_COLORS,
    FACTOR_LABELS,
    RESIDUAL_GROUP,
    SYSTEMS,
    FactorSpace,
    append_best_system,
    define_groups,
    grid_outputs,
    outputs_table,
    plot_influence_map,
    plot_sobol_aggregate,
    sobol_design,
)

# ---------------------------------------------------------------------
# Configurazione (il resto viene da sobol_analysis.py)
# ---------------------------------------------------------------------
OUT_DIR = HERE / "shapley"
MAP_DIR = OUT_DIR / "mappe"
DPI = 140
CALIBRATION_FACTOR_NAME = "calibration_block"

# nomi leggibili dei parametri di calibrazione in figura:
# {"nome_colonna_theta_acc": "Nome in figura", ...}; quelli non elencati
# compaiono con il nome della colonna
CALIBRATION_LABELS = {}

# un colore per parametro di calibrazione, nell'ordine delle colonne di
# theta_acc; scelti diversi da quelli dei parametri tecnologici
CALIBRATION_PALETTE = ["#F781BF", "#984EA3", "#A6D854", "#E5C494", "#8DD3C7", "#FB8072",
                       "#BC80BD", "#FFED6F"]

BAR_TITLE_SIZE = 14
BAR_LABEL_SIZE = 13


# =====================================================================
# 1. STIMATORE
# =====================================================================

def _ranks(X: pd.DataFrame) -> np.ndarray:
    return X.rank(pct=True, method="average").to_numpy(dtype=float)


def _row_ids(X: pd.DataFrame) -> np.ndarray:
    """Stesso id per la stessa riga di theta_acc estratta più volte"""
    return X.groupby(list(X.columns), sort=False).ngroup().to_numpy()


def _nearest_other(points, row_id, exclude_same_row: bool) -> np.ndarray:
    """Indice del campione più vicino diverso da sé (e, per i
    sottoinsiemi stretti, con una riga di theta_acc diversa)"""
    n = len(points)
    own = np.arange(n)
    k = min(n, (np.bincount(row_id).max() if exclude_same_row else 1) + 2)
    _, idx = cKDTree(points).query(points, k=k)
    idx = np.atleast_2d(idx)
    bad = idx == own[:, None]
    if exclude_same_row:
        bad |= row_id[idx] == row_id[:, None]
    if bad.all(axis=1).any():
        raise RuntimeError("Vicino non trovato: troppe righe ripetute")
    return idx[own, np.argmax(~bad, axis=1)]


def shapley_effects(X: pd.DataFrame, Y: np.ndarray) -> tuple:
    """Effetti di Shapley delle colonne di X per ogni colonna di Y.

    X  (n, p) i parametri correlati;  Y (n, n_out) risposte trasformate
    Ritorna (shapley (n_out, p), totale (n_out,) = sum_j shapley)
    """
    R, rid = _ranks(X), _row_ids(X)
    p = R.shape[1]
    fin = np.isfinite(Y)
    with np.errstate(invalid="ignore"):
        V = np.nanvar(Y, axis=0, ddof=1)

    val = {0: np.zeros(Y.shape[1])}
    for size in range(1, p + 1):
        for S in combinations(range(p), size):
            nn = _nearest_other(R[:, list(S)], rid, exclude_same_row=size < p)
            ok = fin & fin[nn]
            d2 = np.where(ok, (Y - np.where(ok, Y[nn], 0.0)) ** 2, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                val[sum(1 << j for j in S)] = 1.0 - 0.5 * d2.sum(0) / ok.sum(0) / V

    full = (1 << p) - 1
    w = [factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)]
    shap = np.zeros((Y.shape[1], p))
    for j in range(p):
        bit = 1 << j
        for m in range(full + 1):
            if not m & bit:
                shap[:, j] += w[bin(m).count("1")] * (val[m | bit] - val[m])
    return shap, val[full]


def st_equivalent(shap: np.ndarray, st_block: np.ndarray) -> np.ndarray:
    """S_T del blocco ripartito fra i parametri in proporzione ai loro
    effetti di Shapley (negativi = rumore = zero)"""
    pos = np.clip(shap, 0.0, None)
    tot = pos.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / shap.shape[1])
    return share * st_block[:, None]


# =====================================================================
# 2. DATI
# =====================================================================

def load_samples():
    """(X_cal, Yt, meta, theta_acc) dai blocchi A e B della cache.
    meta ha V, frazione_valida e fattibile calcolati come nella Sobol"""
    theta_acc = load_theta_acc(SA.THETA_ACC_PATH)
    space = FactorSpace.default(theta_acc)
    all_outputs = grid_outputs(SA.RANGES_NMI, SA.SPEEDS_KT)
    sig = SA._signature(space, all_outputs)

    meta_file = SA.CACHE_DIR / "meta.json"
    if meta_file.exists():
        # cache scritta dalla versione precedente di sobol_analysis.py (firma
        # senza theta_acc): la si aggiorna, senza toccare le valutazioni
        old = json.loads(meta_file.read_text())
        if old.get("signature") == SA._signature(space, all_outputs, include_theta_acc=False):
            SA._write_meta(sig, old["columns"])
    elif SA.LEGACY_CACHE.exists():
        SA._migrate_legacy(sig, space, all_outputs)
    if not meta_file.exists():
        raise SystemExit("Nessuna valutazione in sobol_cache/: lanciare prima sobol_analysis.py")
    meta_json = json.loads(meta_file.read_text())
    if meta_json.get("signature") != sig:
        raise SystemExit("La cache in sobol_cache/ è di un'altra configurazione: "
                         "rilanciare sobol_analysis.py")

    kept = set(meta_json["columns"])
    outputs = [o for o in all_outputs if o.name in kept]
    names = [o.name for o in outputs]
    Y_ab = pd.DataFrame(np.vstack([np.load(SA.CACHE_DIR / "A.npy"),
                                   np.load(SA.CACHE_DIR / "B.npy")]), columns=names)
    meta = outputs_table(outputs)
    if SA.INCLUDE_BEST:
        Y_ab, meta = append_best_system(Y_ab, meta, candidates=SA.BEST_CANDIDATES)

    # A e B dipendono solo da N, seed e numero di fattori
    design = sobol_design(space, SA.SELECTED_FACTORS, N=SA.N_BASE, seed=SA.SEED)
    theta = space.theta_from_unit(design.U[:2 * design.N])
    X_cal = theta[list(theta_acc.columns)].reset_index(drop=True)

    Yt = Y_ab[meta["output"]].to_numpy(dtype=float)
    if SA.TRANSFORM == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            Yt = np.where(Yt > 0, np.log(Yt), np.nan)
    frac = np.isfinite(Yt).mean(axis=0)
    with np.errstate(invalid="ignore"):
        V = np.nanvar(Yt, axis=0, ddof=1)
    meta = meta.assign(V=V, frazione_valida=frac)
    return X_cal, Yt, meta.reset_index(drop=True), theta_acc


def load_sobol(meta: pd.DataFrame) -> tuple:
    """Gli S_T della Sobol (sobol_indices.csv), allineati agli output di
    meta: (ST (n_out, g), nomi dei fattori, fattibile)"""
    f = SA.FIGURE_DIR / "sobol_indices.csv"
    if not f.exists():
        raise SystemExit("sobol_indices.csv non trovato: lanciare prima sobol_analysis.py")
    s = pd.read_csv(f)
    factors = list(pd.unique(s["fattore"]))
    if CALIBRATION_FACTOR_NAME not in factors:
        raise SystemExit(f"'{CALIBRATION_FACTOR_NAME}' non è fra i fattori della Sobol: "
                         "aggiungerlo a SELECTED_FACTORS e rilanciare sobol_analysis.py "
                         "(riusa le valutazioni, costa solo due blocchi)")
    ST = s.pivot(index="output", columns="fattore", values="ST").reindex(meta["output"])[factors]
    S1 = s.pivot(index="output", columns="fattore", values="S1").reindex(meta["output"])
    feas = s.groupby("output")["fattibile"].first().reindex(meta["output"])
    if ST.isna().all(axis=1).any():
        raise SystemExit("sobol_indices.csv non corrisponde agli output della cache: "
                         "rilanciare sobol_analysis.py")
    return (ST.to_numpy(dtype=float), factors, feas.fillna(False).to_numpy(dtype=bool),
            S1[CALIBRATION_FACTOR_NAME].to_numpy(dtype=float))


# =====================================================================
# 3. AGGREGAZIONE (come per la Sobol)
# =====================================================================

def _mask(meta: pd.DataFrame, criteri: dict) -> np.ndarray:
    m = np.ones(len(meta), dtype=bool)
    for col, v in criteri.items():
        m &= meta[col].isin(v if isinstance(v, (list, tuple, set)) else [v]).to_numpy()
    return m


def _weights(meta, M, feas, mask) -> np.ndarray:
    w = meta["V"].to_numpy(dtype=float)
    ok = mask & feas & np.isfinite(w) & np.isfinite(M).all(axis=1)
    return np.where(ok, w, 0.0)


def aggregate_bars(meta, M, names, feas, mask=None) -> pd.DataFrame:
    """Stesse colonne di aggregate_indices: ST (media pesata), ST_min/max"""
    if mask is None:
        mask = meta["sistema"].isin(SYSTEMS).to_numpy()
    w = _weights(meta, M, feas, mask)
    sel = w > 0
    ws, Ms = w[sel], M[sel]
    out = pd.DataFrame({"fattore": names,
                        "ST": (ws[:, None] * Ms).sum(0) / ws.sum(),
                        "ST_min": Ms.min(0), "ST_max": Ms.max(0),
                        "disp_lo": np.nan, "disp_hi": np.nan,
                        "ST_lo": np.nan, "ST_hi": np.nan,
                        "n_punti": int(sel.sum())})
    out = out.sort_values("ST", ascending=False).reset_index(drop=True)
    out.attrs.update(weights="varianza", spread=(0, 100))
    return out


def leader_map(meta, M, names, feas, mask, ranges, speeds, candidates) -> pd.DataFrame:
    """Stesse colonne di influence_map"""
    w = _weights(meta, M, feas, mask)
    R, Vk = meta["range_nmi"].to_numpy(), meta["velocita_kt"].to_numpy()
    names = np.array(names)
    rows = []
    for v in speeds:
        for r in ranges:
            cell = (w > 0) & np.isclose(R, r, rtol=1e-6) & np.isclose(Vk, v, rtol=1e-6)
            base = {"range_nmi": float(r), "velocita_kt": float(v),
                    "n_architetture": int(cell.sum())}
            if not cell.any():
                rows.append({**base, "fattibile": False, "leader": None, "ST_leader": np.nan,
                             "secondo": None, "ST_secondo": np.nan})
                continue
            agg = (w[cell, None] * M[cell]).sum(0) / w[cell].sum()
            o = np.argsort(-np.where(candidates, agg, -np.inf))
            rows.append({**base, "fattibile": True, "leader": names[o[0]],
                         "ST_leader": float(agg[o[0]]), "secondo": names[o[1]],
                         "ST_secondo": float(agg[o[1]])})
    return pd.DataFrame(rows)


# =====================================================================
# 4. FIGURE
# =====================================================================

def _label(p):
    return CALIBRATION_LABELS.get(p, p)


def plot_shapley_bars(agg: pd.DataFrame, colors: dict):
    """Effetti di Shapley dei 6 parametri (baffi min-max fra gli output);
    sommano alla quota diretta della calibrazione, scritta nel titolo"""
    sub = agg.sort_values("shapley", ascending=True).reset_index(drop=True)
    val = sub["shapley"].clip(lower=0.0)
    lo, hi = sub["shapley_min"].clip(0, 1), sub["shapley_max"].clip(0, 1)
    err = np.vstack([np.clip(val - lo, 0, None), np.clip(hi - val, 0, None)])
    fig, ax = plt.subplots(figsize=(10.0, 0.6 * len(sub) + 2.4))
    y = np.arange(len(sub))
    ax.barh(y, val, height=0.62, color=[colors[p] for p in sub["parametro"]], alpha=0.95,
            xerr=err, error_kw=dict(lw=1.2, ecolor="0.25", capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels([_label(p) for p in sub["parametro"]], fontsize=BAR_LABEL_SIZE)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Shapley effect (fraction of variance)", fontsize=15)
    ax.grid(axis="x", alpha=0.25)
    tot = float(agg["totale_calibrazione"].iloc[0])
    ax.set_title("Calibration Parameters: Shapley Effects\n"
                 f"(variance-weighted; sum = {tot:.2f}; whiskers: min-max)",
                 fontsize=BAR_TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labs = [_label(p) for p in corr.columns]
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=11)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman correlation")
    ax.set_title("Accepted Calibration Parameters:\nRank Correlations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5. LANCIO
# =====================================================================

def main():
    pd.set_option("display.width", 200)
    OUT_DIR.mkdir(exist_ok=True)
    MAP_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    X_cal, Yt, meta, theta_acc = load_samples()
    ST_sob, factors, feas, S1_block = load_sobol(meta)
    params = list(X_cal.columns)
    print(f"{len(X_cal)} campioni (blocchi A+B) x {Yt.shape[1]} output; "
          f"parametri di calibrazione: {', '.join(params)}")

    # --- effetti di Shapley e S_T equivalenti
    shap, total = shapley_effects(X_cal, Yt)
    jb = factors.index(CALIBRATION_FACTOR_NAME)
    st_eq = st_equivalent(shap, ST_sob[:, jb])
    print(f"Effetti di Shapley calcolati in {time.time() - t0:.0f}s")

    pd.DataFrame([{"output": meta["output"].iloc[o], "parametro": p,
                   "shapley": shap[o, j], "ST_equivalente": st_eq[o, j],
                   "fattibile": bool(feas[o])}
                  for o in range(len(meta)) for j, p in enumerate(params)]) \
        .to_csv(OUT_DIR / "shapley_indices.csv", index=False)

    # --- aggregato dei soli effetti di Shapley
    m_all = meta["sistema"].isin(SYSTEMS).to_numpy()
    w = _weights(meta, shap, feas, m_all)
    sel = w > 0
    agg_sh = pd.DataFrame({"parametro": params,
                           "shapley": (w[sel, None] * shap[sel]).sum(0) / w[sel].sum(),
                           "shapley_min": shap[sel].min(0), "shapley_max": shap[sel].max(0)})
    agg_sh = agg_sh.sort_values("shapley", ascending=False).reset_index(drop=True)
    agg_sh["totale_calibrazione"] = float((w[sel] * total[sel]).sum() / w[sel].sum())
    agg_sh.to_csv(OUT_DIR / "shapley_aggregate.csv", index=False)

    corr = theta_acc.corr(method="spearman")
    corr.to_csv(OUT_DIR / "calibration_correlation.csv")

    s1b = float(np.nansum(w * S1_block) / w.sum())
    stb = float(np.nansum(w * ST_sob[:, jb]) / w.sum())
    print("\n" + "=" * 78)
    print("EFFETTI DI SHAPLEY AGGREGATI (tutte le architetture, pesi = Var(log E))")
    print("=" * 78)
    print(agg_sh.round(3).to_string(index=False))
    print(f"\nSomma degli effetti di Shapley:     {agg_sh['totale_calibrazione'].iloc[0]:.3f}")
    print(f"S_1[calibration_block] (Sobol):     {s1b:.3f}   <- deve essere vicino alla somma")
    print(f"S_T[calibration_block] (Sobol):     {stb:.3f}   <- ripartito fra i 6 nelle figure")
    print(f"Quota di interazione del blocco:    {stb - s1b:.3f}")

    # --- figure della Sobol con i 6 parametri al posto del blocco
    names = [f for f in factors if f != CALIBRATION_FACTOR_NAME] + params
    M = np.column_stack([np.delete(ST_sob, jb, axis=1), st_eq])
    cal_colors = {p: CALIBRATION_PALETTE[i % len(CALIBRATION_PALETTE)] for i, p in enumerate(params)}
    FACTOR_COLORS.update(cal_colors)                       # stessi colori in barre e mappe
    FACTOR_LABELS.update({p: _label(p) for p in params})

    agg = aggregate_bars(meta, M, names, feas)
    agg.to_csv(OUT_DIR / "sobol_aggregate.csv", index=False)
    ax = plot_sobol_aggregate(agg, error="range",
                              title="Aggregated Sobol Total Indices\n"
                                    "(calibration block split by Shapley effects)")
    _save(ax.figure, OUT_DIR / "sobol_aggregate.png")

    ranges = sorted(meta["range_nmi"].unique())
    speeds = sorted(meta["velocita_kt"].unique())
    candidates = np.array([SA.LEADER_INCLUDE_RESIDUAL or n != RESIDUAL_GROUP for n in names])
    maps = []
    for g in define_groups(meta, include_best=SA.INCLUDE_BEST):
        mp = leader_map(meta, M, names, feas, _mask(meta, g["criteri"]), ranges, speeds,
                        candidates)
        maps.append(mp.assign(livello=g["livello"], gruppo=g["nome"], file=g["slug"]))
        ax = plot_influence_map(mp, ranges, speeds, g["nome"],
                                title="Most Influential Parameter\n(calibration block split by Shapley effects)")
        _save(ax.figure, MAP_DIR / f"{g['slug']}.png")
    pd.concat(maps, ignore_index=True).to_csv(OUT_DIR / "influence_maps.csv", index=False)

    _save(plot_shapley_bars(agg_sh, cal_colors), OUT_DIR / "shapley_aggregate.png")
    _save(plot_correlation(corr), OUT_DIR / "calibration_correlation.png")
    print(f"\nRisultati in {OUT_DIR} ({len(maps)} mappe in {MAP_DIR.name}/)")


if __name__ == "__main__":
    main()