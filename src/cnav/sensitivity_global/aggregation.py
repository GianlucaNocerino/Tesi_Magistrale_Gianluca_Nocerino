"""
Come si aggregano indici di Sobol di output diversi.

Il problema
-----------
Ogni output (architettura a, nodo (R, V)) ha i suoi indici S_T,i(a,R,V),
frazioni della SUA varianza. Una media aritmetica di frazioni di
varianze diverse non è la frazione di nessuna varianza: dipende da
quanti punti si sono assegnati a ciascun sistema e tratta allo stesso
modo un punto dove l'intensity è quasi certa e uno dove varia del 100%.

La scelta di default: media pesata con la varianza
--------------------------------------------------
Per un insieme G di output,

    S_T,i^G = sum_{k in G} V_k * S_T,i^(k) / sum_{k in G} V_k

È l'indice di Sobol generalizzato di un output vettoriale (Lamboni et
al. 2011; Gamboa, Janon, Klein, Lagnoux 2014): la frazione della
varianza TOTALE del vettore (E_k)_{k in G} attribuibile al fattore i.
Quindi è ancora una frazione di varianza, ha una definizione citabile,
e i punti contano tanto quanta incertezza portano.

Funziona perché si decompone log(E): V_k = Var(log E_k) ≈ CV_k^2 è
adimensionale, e sistemi con intensity di ordini di grandezza diversi
(fuel cell a decine di MJ/pax/nmi, combustione a poche unità) pesano
per la loro incertezza RELATIVA e non per la loro scala. Senza log la
pesatura con la varianza farebbe decidere tutto al sistema più
"grosso" e non andrebbe usata.

Alternative (parametro weights):

  "uniforme"              media semplice: ogni punto fattibile conta
                          uguale. Più facile da spiegare, meno fondata
  "probabilita_migliore"  pesi = probabilità che quell'architettura sia
                          la migliore in quel nodo (stimata dagli stessi
                          campioni A+B). Risponde alla domanda
                          decisionale: "chi governa l'incertezza dei
                          sistemi che contano davvero qui?"

Un'architettura infattibile in un nodo (più di nan_tol di NaN) non
entra: altrimenti la dimensione del dominio di fattibilità diventerebbe
un peso implicito.

La griglia è regolare in log(R) e in V, quindi aggregare su tutti i
nodi equivale a mediare sull'area della mappa così come è disegnata.

Le mappe
--------
Gli indici di Sobol sono proprietà deterministiche del modello e delle
PDF dei parametri: la mappa riporta, nodo per nodo, il parametro con
S_T aggregato massimo e nient'altro. L'unica incertezza è quella di
stima (N finito), che si controlla una volta per tutte con la prova di
convergenza (sobol_convergence) e non entra nelle figure.

Le barre di errore del grafico aggregato
----------------------------------------
Di default sono la DISPERSIONE di S_T nel piano (percentili pesati
10-90 fra i punti aggregati): quanto il peso del parametro cambia da
una missione all'altra. In alternativa, l'intervallo bootstrap al 95%
dell'aggregato (errore di stima), possibile perché il bootstrap usa gli
stessi ricampionamenti per tutti gli output
"""
from typing import Optional

import numpy as np
import pandas as pd

from .outputs import BEST_SYSTEM, PROPULSORS, SYSTEMS
from .sobol import RESIDUAL_GROUP, SobolField, sobol_field

__all__ = [
    "WEIGHT_SCHEMES",
    "define_groups",
    "best_probability",
    "aggregate_indices",
    "influence_map",
    "best_system_map",
    "sobol_convergence",
]

WEIGHT_SCHEMES = ("varianza", "uniforme", "probabilita_migliore")


# ---------------------------------------------------------------------
# Gruppi di aggregazione
# ---------------------------------------------------------------------

def define_groups(meta: pd.DataFrame, include_best: bool = True) -> list:
    """I livelli di aggregazione, dal più aggregato al meno:

      complessivo   tutte le architetture
      propulsore    tutti i fan / tutte le eliche
      sistema       fan + elica dello stesso sistema
      architettura  una per (sistema, propulsore)
      migliore      la pseudo-architettura E_best (se presente)

    Ritorna una lista di dict {livello, nome, slug, criteri}, dove
    criteri va passato a SobolField.mask(**criteri)
    """
    systems = [s for s in SYSTEMS if s in set(meta["sistema"])]
    props = [p for p in PROPULSORS if p in set(meta["propulsore"])]

    def slug(s):
        s = s.lower().replace("+", " ").replace("(", " ").replace(")", " ").replace("-", " ")
        return "_".join(s.split())

    g = [{"livello": "complessivo", "nome": "All architectures",
          "criteri": {"sistema": systems}}]
    for p in props:
        g.append({"livello": "propulsore", "nome": f"All {p} architectures",
                  "criteri": {"sistema": systems, "propulsore": p}})
    for s in systems:
        g.append({"livello": "sistema", "nome": f"{s} (fan + propeller)",
                  "criteri": {"sistema": s}})
    for s in systems:
        for p in props:
            g.append({"livello": "architettura", "nome": f"{s} ({p})",
                      "criteri": {"sistema": s, "propulsore": p}})
    if include_best and BEST_SYSTEM in set(meta["sistema"]):
        g.append({"livello": "migliore", "nome": "Best architecture (min intensity)",
                  "criteri": {"sistema": BEST_SYSTEM}})
    for i, d in enumerate(g):
        d["slug"] = f"{i:02d}_{d['livello']}_{slug(d['nome'])}"
    return g


# ---------------------------------------------------------------------
# Probabilità di essere il migliore
# ---------------------------------------------------------------------

def best_probability(design, Y: pd.DataFrame, meta: pd.DataFrame,
                     candidates: Optional[list] = None) -> pd.Series:
    """P(architettura migliore nel suo nodo), sul campione onesto A+B.

    Migliore = intensity minima fra le architetture fattibili in quel
    campione; un campione in cui nessuna è fattibile non assegna
    vittorie. È la stessa quantità della mappa "most likely best" (se
    quella usa l'intensity come criterio): plottarla con
    best_system_map serve anche da controllo di coerenza fra le due
    analisi. Le colonne non valutate (scartate dal prescreen) valgono 0
    """
    m = meta[(meta["sistema"] != BEST_SYSTEM) & meta["output"].isin(Y.columns)]
    if candidates is not None:
        m = m[m["architettura"].isin(candidates)]
    rows = np.r_[0:2 * design.N]                         # blocchi A e B
    P = pd.Series(0.0, index=meta["output"])
    for _, cell in m.groupby(["range_nmi", "velocita_kt"], sort=False):
        names = cell["output"].tolist()
        v = Y[names].to_numpy()[rows]
        fin = np.isfinite(v)
        any_ok = fin.any(axis=1)
        win = np.argmin(np.where(fin, v, np.inf), axis=1)
        counts = np.bincount(win[any_ok], minlength=len(names))
        P.loc[names] = counts / len(rows)
    return P


# ---------------------------------------------------------------------
# Pesi
# ---------------------------------------------------------------------

def _weights(fld: SobolField, scheme: str, p_best: Optional[pd.Series]) -> tuple:
    """(w, w_boot): pesi puntuali (n_out,) e per replica (n_out, B).
    Zero per gli output non fattibili"""
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"weights deve essere uno fra {WEIGHT_SCHEMES}")
    feas = fld.outputs["fattibile"].to_numpy()
    B = fld.n_boot
    if scheme == "varianza":
        w = fld.outputs["V"].to_numpy(dtype=float)
        wb = fld.V_boot if B else np.zeros((len(w), 0))
    elif scheme == "uniforme":
        w = np.ones(len(feas))
        wb = np.ones((len(w), B))
    else:
        if p_best is None:
            raise ValueError("weights='probabilita_migliore' richiede p_best "
                             "(best_probability)")
        w = p_best.reindex(fld.outputs["output"]).fillna(0.0).to_numpy(dtype=float)
        wb = np.repeat(w[:, None], B, axis=1)
    w = np.where(feas & np.isfinite(w), w, 0.0)
    wb = np.where(feas[:, None] & np.isfinite(wb), wb, 0.0)
    return w, wb


def _weighted_quantile(x, w, q):
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.full(len(q), np.nan)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(q) / 100.0, c, x)


def _combine(ST, STb, w, wb):
    """Media pesata su un insieme di output: (g,) e (g, B)"""
    agg = (w[:, None] * ST).sum(0) / w.sum()
    if STb.shape[-1]:
        valid = np.isfinite(STb)
        num = np.where(valid, wb[:, None, :] * STb, 0.0).sum(0)
        den = np.where(valid, wb[:, None, :], 0.0).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            agg_b = num / den
    else:
        agg_b = np.zeros((ST.shape[1], 0))
    return agg, agg_b


# ---------------------------------------------------------------------
# Indici aggregati (il grafico a barre)
# ---------------------------------------------------------------------

def aggregate_indices(fld: SobolField, mask: Optional[np.ndarray] = None,
                      weights: str = "varianza", p_best: Optional[pd.Series] = None,
                      spread: tuple = (10, 90)) -> pd.DataFrame:
    """S_T aggregato sugli output in mask (default: tutte le
    architetture vere, esclusa la pseudo-architettura E_best).

    Colonne:
      ST              l'indice aggregato (media pesata)
      ST_lo, ST_hi    intervallo bootstrap al 95% dell'aggregato
      disp_lo/hi      percentili pesati `spread` di S_T fra i punti
                      aggregati: quanto il fattore cambia peso nel piano
      ST_min/max      il valore più piccolo e più grande di S_T fra tutti
                      gli output aggregati (architetture x punti)
      quota_leader    frazione (pesata) dei punti in cui il fattore è
                      il più influente
      n_punti         output fattibili aggregati
    """
    if mask is None:
        mask = fld.mask(sistema=SYSTEMS)
    w, wb = _weights(fld, weights, p_best)
    sel = mask & (w > 0)
    if not sel.any():
        raise ValueError("Nessun output fattibile con peso positivo nel gruppo")

    ST, STb, ws, wbs = fld.ST[sel], fld.ST_boot[sel], w[sel], wb[sel]
    agg, agg_b = _combine(ST, STb, ws, wbs)
    leader = np.nanargmax(ST, axis=1)

    rows = []
    for j, name in enumerate(fld.group_names):
        if agg_b.shape[-1]:
            lo, hi = np.nanpercentile(agg_b[j], [2.5, 97.5])
        else:
            lo = hi = np.nan
        dlo, dhi = _weighted_quantile(ST[:, j], ws, spread)
        rows.append({"fattore": name, "ST": agg[j], "ST_lo": lo, "ST_hi": hi,
                     "disp_lo": dlo, "disp_hi": dhi,
                     "ST_min": float(np.nanmin(ST[:, j])),
                     "ST_max": float(np.nanmax(ST[:, j])),
                     "quota_leader": ws[leader == j].sum() / ws.sum(),
                     "n_punti": int(sel.sum())})
    out = pd.DataFrame(rows).sort_values("ST", ascending=False).reset_index(drop=True)
    out.attrs.update(weights=weights, spread=spread)
    return out


# ---------------------------------------------------------------------
# Mappe del parametro più influente
# ---------------------------------------------------------------------

def influence_map(fld: SobolField, mask: np.ndarray, ranges, speeds,
                  weights: str = "varianza", p_best: Optional[pd.Series] = None,
                  include_residual: bool = True) -> pd.DataFrame:
    """Per ogni nodo (R, V) della griglia: il fattore con S_T aggregato
    massimo fra le architetture del gruppo (mask) fattibili in quel nodo.

    Colonne: range_nmi, velocita_kt, fattibile, leader, ST_leader,
    secondo, ST_secondo, n_architetture. Il secondo e il suo S_T sono
    solo in tabella (influence_maps.csv): servono a vedere dove il
    primato è netto e dove due parametri si equivalgono.

    In un nodo con una sola architettura fattibile i pesi non contano;
    se tutte le architetture fattibili hanno peso zero (succede con
    "probabilita_migliore" dove nessuna vince mai) si ripiega sulla
    media uniforme, così le mappe per singola architettura restano
    complete.

    include_residual=False esclude "others" dalla corsa al primo posto
    (ma non dal denominatore: gli indici restano frazioni della
    varianza vera). Di default è incluso: se dove l'incertezza non è
    tecnologica la mappa lo dice, è un risultato
    """
    w, _ = _weights(fld, weights, p_best)
    feas = fld.outputs["fattibile"].to_numpy()
    cand = np.array([include_residual or nm != RESIDUAL_GROUP for nm in fld.group_names])
    names = np.array(fld.group_names)
    R = fld.outputs["range_nmi"].to_numpy()
    V = fld.outputs["velocita_kt"].to_numpy()
    sel_all = mask & feas

    rows = []
    for v in speeds:
        for r in ranges:
            cell = sel_all & np.isclose(R, r, rtol=1e-6) & np.isclose(V, v, rtol=1e-6)
            base = {"range_nmi": float(r), "velocita_kt": float(v),
                    "n_architetture": int(cell.sum())}
            if not cell.any():
                rows.append({**base, "fattibile": False, "leader": None,
                             "ST_leader": np.nan,
                             "secondo": None, "ST_secondo": np.nan})
                continue
            wc = w[cell]
            if wc.sum() <= 0:
                wc = np.ones_like(wc)
            agg = (wc[:, None] * fld.ST[cell]).sum(0) / wc.sum()
            score = np.where(cand, agg, -np.inf)
            order = np.argsort(-score)
            lead = order[0]
            sec = order[1] if cand.sum() > 1 else order[0]
            rows.append({**base, "fattibile": True, "leader": names[lead],
                         "ST_leader": float(agg[lead]),
                         "secondo": names[sec], "ST_secondo": float(agg[sec])})
    return pd.DataFrame(rows)


def best_system_map(p_best: pd.Series, meta: pd.DataFrame, ranges, speeds) -> pd.DataFrame:
    """La mappa "most likely best" ricostruita dagli stessi campioni
    (stesse colonne di influence_map, con l'architettura al posto del
    fattore). Serve da controllo: deve somigliare a quella dell'analisi
    di propagazione"""
    m = meta[meta["sistema"] != BEST_SYSTEM].set_index("output")
    m = m.assign(P=p_best.reindex(m.index).fillna(0.0))
    rows = []
    for v in speeds:
        for r in ranges:
            c = m[np.isclose(m["range_nmi"], r, rtol=1e-6) & np.isclose(m["velocita_kt"], v, rtol=1e-6)]
            if c.empty or c["P"].sum() <= 0:
                rows.append({"range_nmi": float(r), "velocita_kt": float(v), "fattibile": False,
                             "leader": None, "P_best": np.nan})
                continue
            top = c["P"].idxmax()
            rows.append({"range_nmi": float(r), "velocita_kt": float(v), "fattibile": True,
                         "leader": c.loc[top, "architettura"], "P_best": float(c["P"].max())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Convergenza
# ---------------------------------------------------------------------

def sobol_convergence(design, Y: pd.DataFrame, meta: pd.DataFrame,
                      sizes: Optional[list] = None, transform: Optional[str] = "log",
                      nan_tol: float = 0.10, weights: str = "varianza",
                      p_best: Optional[pd.Series] = None) -> pd.DataFrame:
    """S_T aggregato (gruppo complessivo) ricalcolato sulle prime n
    righe di ogni blocco, n = 64, 128, ..., N. Nessuna valutazione in
    più. Le curve devono appiattirsi e smettere di incrociarsi"""
    if sizes is None:
        sizes, m = [], 64
        while m <= design.N:
            sizes.append(m)
            m *= 2
        if not sizes or sizes[-1] != design.N:
            sizes.append(design.N)
    parts = []
    meta_i = meta.set_index("output")
    for n in sizes:
        f = sobol_field(design, Y, meta_i, n_boot=0, n=int(n), transform=transform,
                        nan_tol=nan_tol)
        agg = aggregate_indices(f, weights=weights, p_best=p_best)
        parts.append(agg[["fattore", "ST"]].assign(N=int(n)))
    return pd.concat(parts, ignore_index=True)
