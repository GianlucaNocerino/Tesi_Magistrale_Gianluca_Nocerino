"""
Grafici della sensibilità globale.

Perché solo S_T
---------------
Le mappe e il grafico aggregato mostrano l'indice TOTALE:

  - il modello è fortemente non additivo (il sizing è un punto fisso,
    ogni parametro rientra nel MTOW e da lì in tutto il resto): S_1
    ignora proprio quella parte di influenza
  - S_T risponde alla domanda "su quale parametro tecnologico conviene
    ridurre l'incertezza / quale si può fissare senza perdere niente"
    (factor fixing, Saltelli 2008): S_T ≈ 0 è l'unica condizione
    sufficiente per fissare un fattore
  - è non negativo e più stabile di S_1 a parità di N

S_1 e S_T - S_1 restano nella tabella tidy (sobol_indices.csv): dove
servono si citano in tabella, non in figura.

Le figure sono in inglese come il resto della tesi; nomi leggibili dei
fattori in FACTOR_LABELS, colori fissi in FACTOR_COLORS (lo stesso
fattore ha lo stesso colore in tutte le mappe e nelle barre).
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

from .sobol import RESIDUAL_GROUP

__all__ = [
    "FACTOR_LABELS",
    "FACTOR_COLORS",
    "ARCHITECTURE_COLORS",
    "factor_label",
    "plot_sobol_aggregate",
    "plot_influence_map",
    "plot_best_system_map",
    "plot_sobol_convergence",
]

FACTOR_LABELS = {
    "e_battery_Wh_per_kg": "Battery specific energy",
    "fuel_cell_specific_power_kW_per_kg": "Fuel cell specific power",
    "eta_fuel_cell": "Fuel cell efficiency",
    "gamma_tank": "LH$_2$ tank gravimetric index",
    "eta_motor": "Electric motor efficiency",
    "eta_p_propeller": "Propeller efficiency",
    "eta_p_fan": "Fan efficiency",
    "electricity": "Electricity WtT efficiency",
    "liquid_hydrogen": "LH$_2$ production efficiency",
    "e_saf": "e-SAF production efficiency",
    "oew_fan_b": "OEW regression (fan, b)",
    "oew_fan_r_pivot": "OEW regression (fan, r$_{pivot}$)",
    "oew_prop_b": "OEW regression (prop, b)",
    "oew_prop_r_pivot": "OEW regression (prop, r$_{pivot}$)",
    "calibration_block": "Calibration block",
    RESIDUAL_GROUP: "Others",
}

# colori fissi per fattore (palette qualitativa ad alto contrasto,
# distinguibile anche in scala di grigi a coppie adiacenti); "others"
# sempre grigio, perché non è un parametro ma "tutto il resto"
FACTOR_COLORS = {
    "e_battery_Wh_per_kg": "#4477AA",
    "fuel_cell_specific_power_kW_per_kg": "#EE7733",
    "eta_fuel_cell": "#CC3311",
    "gamma_tank": "#228833",
    "eta_motor": "#AA3377",
    "eta_p_propeller": "#33BBEE",
    "eta_p_fan": "#009988",
    "electricity": "#CCBB44",
    "liquid_hydrogen": "#66CCEE",
    "e_saf": "#994F00",
    "oew_fan_b": "#882255",
    "oew_fan_r_pivot": "#DDAA33",
    "oew_prop_b": "#117733",
    "oew_prop_r_pivot": "#332288",
    "calibration_block": "#000000",
    RESIDUAL_GROUP: "#9E9E9E",
}

ARCHITECTURE_COLORS = {
    "Battery-electric (propeller)": "#4477AA",
    "Battery-electric (fan)": "#1F3F66",
    "Hydrogen fuel cell (propeller)": "#EE7733",
    "Hydrogen fuel cell (fan)": "#994411",
    "Hydrogen combustion (propeller)": "#66CC88",
    "Hydrogen combustion (fan)": "#228833",
    "e-SAF combustion (propeller)": "#CC99BB",
    "e-SAF combustion (fan)": "#AA3377",
}

# Stile delle mappe "Most Influential Technological Parameter": tutte le
# dimensioni in un posto solo, da ritoccare qui e rilanciare sobol_figures.py
MAP_FIGSIZE = (12.0, 8.5)      # pollici; l'altezza include la legenda sotto
MAP_TITLE_SIZE = 18
MAP_LABEL_SIZE = 18            # "Range [nmi]", "Cruise Speed [kt]"
MAP_TICK_SIZE = 18             # numerazione degli assi
MAP_LEGEND_SIZE = 15
MAP_LEGEND_NCOL = 2            # colonne della legenda sotto il grafico

_FALLBACK = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def factor_label(name: str) -> str:
    return FACTOR_LABELS.get(name, name)


def _color(name: str, table: dict) -> str:
    if name in table:
        return table[name]
    return _FALLBACK[sum(map(ord, name)) % len(_FALLBACK)]   # deterministico


def _edges(c, log: bool):
    c = np.asarray(c, dtype=float)
    x = np.log10(c) if log else c
    if len(x) == 1:
        e = np.array([x[0] - 0.5, x[0] + 0.5])
    else:
        mid = 0.5 * (x[1:] + x[:-1])
        e = np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]
    return 10 ** e if log else e


# ---------------------------------------------------------------------
# Barre aggregate
# ---------------------------------------------------------------------

def plot_sobol_aggregate(agg: pd.DataFrame, title: Optional[str] = None,
                         error: str = "range", ax=None):
    """S_T aggregato, barre orizzontali ordinate.

    error="range"        (default) baffi dal valore più piccolo al più
                         grande di S_T fra tutti gli output aggregati
                         (ogni architettura in ogni punto della griglia)
    error="dispersione"  (default) baffi = percentili pesati di S_T fra
                         i punti del piano (agg.attrs['spread'], 10-90):
                         quanto il peso del fattore cambia nel dominio.
                         È l'informazione che la media nasconde
    error="bootstrap"    baffi = intervallo al 95% dell'aggregato
                         (incertezza di campionamento; piccolo)
    """
    cols = {"range": ("ST_min", "ST_max"), "dispersione": ("disp_lo", "disp_hi"),
            "bootstrap": ("ST_lo", "ST_hi")}
    if error not in cols:
        raise ValueError(f"error deve essere uno fra {list(cols)}")
    sub = agg.sort_values("ST", ascending=True).reset_index(drop=True)
    lo, hi = cols[error]
    # S_T vero sta in [0, 1]: valori appena fuori sono rumore di stima e
    # vengono troncati nel grafico (nel CSV restano quelli calcolati)
    lo_v = sub[lo].clip(lower=0.0, upper=1.0)
    hi_v = sub[hi].clip(lower=0.0, upper=1.0)
    err = np.vstack([np.clip(sub["ST"] - lo_v, 0, None),
                     np.clip(hi_v - sub["ST"], 0, None)])

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10.0, 0.55 * len(sub) + 2.4))

    y = np.arange(len(sub))
    colors = [_color(nm, FACTOR_COLORS) for nm in sub["fattore"]]
    ax.barh(y, sub["ST"], height=0.62, color=colors, alpha=0.9,
            xerr=err, error_kw=dict(lw=1.2, ecolor="0.25", capsize=3))
    ax.axvline(0.0, color="0.3", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([factor_label(nm) for nm in sub["fattore"]], fontsize=13)
    ax.set_xlabel(r"Total-effect Sobol index $S_{T_i}$ (fraction of variance)", fontsize=15)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.25)

    w = agg.attrs.get("weights", "")
    sp = agg.attrs.get("spread", (10, 90))
    wtxt = {"varianza": "variance-weighted", "uniforme": "unweighted",
            "probabilita_migliore": "weighted by P(best)"}.get(w, w)
    etxt = {"range": "whiskers: min-max over all architectures and design points",
            "dispersione": f"whiskers: {sp[0]}th-{sp[1]}th percentile across the domain",
            "bootstrap": "whiskers: 95% bootstrap CI"}[error]
    ax.set_title((title or "Aggregated Sobol Total Indices"),
                 fontsize=18, fontweight="bold")
    if created:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------
# Mappe
# ---------------------------------------------------------------------

def _draw_category_map(ax, mp: pd.DataFrame, ranges, speeds, colors, labels,
                       legend_title: Optional[str] = None, legend_loc: str = "outside",
                       label_size: float = 20, tick_size: float = 14,
                       legend_size: float = 13, legend_ncol: int = 2):
    """Disegno comune: una cella per nodo, colore pieno = categoria in
    testa; celle non fattibili tratteggiate in grigio"""
    re, ve = _edges(ranges, log=True), _edges(speeds, log=False)
    ri = {float(r): i for i, r in enumerate(ranges)}
    vi = {float(v): j for j, v in enumerate(speeds)}

    patches, faces, hatched = [], [], []
    present = []
    for row in mp.itertuples():
        i, j = ri[float(row.range_nmi)], vi[float(row.velocita_kt)]
        rect = Rectangle((re[i], ve[j]), re[i + 1] - re[i], ve[j + 1] - ve[j])
        if not row.fattibile:
            hatched.append(rect)
            continue
        patches.append(rect)
        faces.append(to_rgb(colors(row.leader)))
        if row.leader not in present:
            present.append(row.leader)

    ax.add_collection(PatchCollection(patches, facecolors=faces, edgecolors=faces,
                                      linewidths=0.3))
    if hatched:
        ax.add_collection(PatchCollection(hatched, facecolor="0.93", edgecolor="0.75",
                                          hatch="///", linewidths=0.0))

    ax.set_xscale("log")
    ax.set_xlim(re[0], re[-1])
    ax.set_ylim(ve[0], ve[-1])
    ax.set_xlabel("Range [nmi]", fontsize=label_size)
    ax.set_ylabel("Cruise Speed [kt]", fontsize=label_size)
    ax.tick_params(which="major", labelsize=tick_size)

    order = [c for c in labels if c in present] + [c for c in present if c not in labels]
    handles = [Patch(facecolor=colors(c), label=lab) for c, lab in
               ((c, labels.get(c, c) if isinstance(labels, dict) else c) for c in order)]
    if hatched:
        handles.append(Patch(facecolor="0.93", edgecolor="0.6", hatch="///",
                             label="Infeasible / censored"))
    if legend_loc == "below":
        # sotto l'asse x, centrata; l'offset è in frazioni dell'altezza
        # degli assi e scala con la dimensione dell'etichetta dell'asse
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.07 - 0.004 * label_size),
                  ncol=legend_ncol, fontsize=legend_size, title=legend_title,
                  title_fontsize=legend_size, frameon=False,
                  columnspacing=1.6, handlelength=1.6, handleheight=1.0)
    elif legend_loc == "outside":
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=13, title=legend_title, title_fontsize=13, frameon=False)
    else:
        ax.legend(handles=handles, loc=legend_loc, fontsize=13, framealpha=0.8,
                  title=legend_title)
    return ax


def plot_influence_map(mp: pd.DataFrame, ranges, speeds, group_name: str,
                       weights: str = "varianza", ax=None,
                       title: str = "Most Influential Technological Parameter",
                       legend_title: str = r"Largest $S_{T_i}$",
                       colors: Optional[dict] = None, labels: Optional[dict] = None):
    """Mappa del parametro tecnologico più influente (S_T massimo) al
    variare di range e velocità: una cella per nodo della griglia,
    colore = parametro in testa. Deterministica: nessuna sfumatura e
    nessun contour"""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=MAP_FIGSIZE)
    _draw_category_map(ax, mp, ranges, speeds,
                       colors=lambda c: _color(c, colors or FACTOR_COLORS),
                       labels=labels if labels is not None else FACTOR_LABELS, legend_loc="below",
                       label_size=MAP_LABEL_SIZE, tick_size=MAP_TICK_SIZE,
                       legend_size=MAP_LEGEND_SIZE, legend_ncol=MAP_LEGEND_NCOL,
                       legend_title=legend_title)
    wtxt = {"varianza": "variance-weighted", "uniforme": "unweighted",
            "probabilita_migliore": "weighted by P(best)"}.get(weights, weights)
    n_arch = mp["n_architetture"].max() if "n_architetture" in mp else 1
    sub = group_name if n_arch <= 1 else f"{group_name} — {wtxt}"
    ax.set_title(f"{title}\n{sub}",
                 fontsize=MAP_TITLE_SIZE, fontweight="bold")
    if created:
        plt.tight_layout()
    return ax


def plot_best_system_map(mp: pd.DataFrame, ranges, speeds, ax=None):
    """Controllo di coerenza: la mappa "most likely best" ricostruita dai
    campioni della Sobol (blocchi A+B)"""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(13.0, 6.8))
    _draw_category_map(ax, mp, ranges, speeds,
                       colors=lambda c: _color(c, ARCHITECTURE_COLORS),
                       labels=list(ARCHITECTURE_COLORS))
    ax.set_title("Most Likely Best Carbon-Neutral Propulsion System\n"
                 "(check: rebuilt from the Sobol A+B samples)",
                 fontsize=18, fontweight="bold")
    if created:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------
# Convergenza
# ---------------------------------------------------------------------

def plot_sobol_convergence(conv: pd.DataFrame, ax=None):
    """S_T aggregato in funzione di N, una curva per fattore"""
    piv = conv.pivot_table(index="N", columns="fattore", values="ST")
    piv = piv[piv.iloc[-1].sort_values(ascending=False).index]
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10.0, 6.0))
    for nm in piv.columns:
        ax.plot(piv.index, piv[nm], marker="o", ms=4, lw=2.0,
                color=_color(nm, FACTOR_COLORS), label=factor_label(nm))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (base sample size)", fontsize=16)
    ax.set_ylabel(r"Aggregated $S_{T_i}$", fontsize=16)
    ax.set_title("Convergence of the Aggregated Total Indices", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    if created:
        plt.tight_layout()
    return ax