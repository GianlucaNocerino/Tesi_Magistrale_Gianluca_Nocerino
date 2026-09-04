"""
Screening di Morris (effetti elementari).

L'idea: invece di stimare indici di varianza (costoso), si misurano
tanti rapporti incrementali sparsi per lo spazio dei fattori,

    EE_i = [ Y(u_1, ..., u_i + delta, ..., u_k) - Y(u) ] / delta

e se ne guardano media e dispersione. Ogni traiettoria costa k+1
valutazioni e produce un EE per ciascun fattore, quindi r traiettorie
costano r*(k+1) valutazioni contro le N*(k+2) di Sobol.

Le tre statistiche, per fattore:

  mu       media degli EE con segno. Vicino a zero può voler dire
           "ininfluente" oppure "effetto che cambia segno": da solo non
           basta, ed è la ragione per cui esiste mu*
  mu*      media dei |EE|. Sarebbe la misura di influenza complessiva, ed è
           quella su cui si ordina lo screening
  sigma    deviazione standard degli EE. Alta = l'effetto del fattore
           dipende da dove ci si trova nello spazio, cioè il fattore è
           coinvolto in interazioni o entra in modo fortemente
           nonlineare

Morris ordina, non quantifica: dice chi tenere, non quanta varianza
spiega. La quantificazione è compito di Sobol, sul sottoinsieme scelto.

SCALA
-----
Gli EE sono calcolati sui QUANTILI (i fattori vivono in [0,1], vedi
factors.py), non sui valori fisici. Sono quindi già adimensionali e
confrontabili fra parametri con unità diverse: mu* si legge come
"variazione di Y attraversando l'intera PDF del parametro". Restano
invece nelle unità di Y, per cui mu* di output diversi non si
confronta fra loro, e infatti la tabella è per output

I GRAFICI A NUBE
----------------
Ogni punto della nube è un fattore. In ascissa mu*, la media dei |EE|: quanto il
fattore sposta Y attraversando tutta la sua PDF. In ordinata sigma, la
dispersione degli EE: quanto quello spostamento dipende da dove ci si
trova nello spazio, cioè interazioni e nonlinearità.

  vicino all'origine        ininfluente, si può togliere dalla Sobol
  mu* alto, sigma basso     effetto forte e sostanzialmente additivo
  sigma alto                effetto che dipende dagli altri fattori:
                            da tenere anche a mu* medio, perchè è
                            proprio dove S_Ti - S_i sarà diverso da
                            zero, cioe' il risultato che la guideline
                            chiede alla Sobol

Il filtro è, quindi, la distanza dall'origine, d = sqrt(mu*^2 +
sigma^2), non il solo mu*: è l'unico criterio che tratta allo stesso
modo un fattore forte e additivo e un fattore medio ma fortemente
interagente.

La normalizzazione, e perchè serve
-----------------------------------
mu* e sigma sono nelle unità di Y. Le colonne di Y sono tutte
intensity (MJ/pax/nmi), ma non dello stesso ordine di grandezza: il
fuel cell con fan sta a decine, la combustione a idrogeno a poche
unità. Sommare le nuvole grezze significherebbe far decidere il
ranking cumulativo al punto operativo più "grosso".

Prima di aggregare, ogni output viene quindi diviso per il proprio mu*
massimo. Il fattore più influente di ogni punto finisce a mu* = 1 e la
lettura diventa relativa ("quanto conta rispetto al dominante di quel
punto"). mu* e sigma sono divisi per lo stesso numero, quindi il
rapporto sigma/mu* non cambia e la geometria della nube (chi interagisce
e chi no) è esattamente quella dei dati grezzi. Sarebbe la stessa
normalizzazione di ranking_table, qui sotto.

Attenzione a cosa non dice il grafico cumulativo: è una media su punti
operativi scelti da noi, quindi pesa i sistemi in proporzione a quante
missioni gli abbiamo assegnato. Va letto insieme alla colonna
n_selezionato, che dice in quanti punti quel fattore ha superato la
soglia: un fattore dominante in un solo punto è un risultato diverso
da un fattore mediamente rilevante ovunque, ed è la stessa distinzione
che nell'analisi locale faceva n_relevant/n_conditions_total
"""
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

__all__ = [
    "morris_trajectories", "elementary_effects", "morris_indices", "ranking_table",
    "morris_cloud_table", "aggregate_morris_cloud", "select_factors",
    "plot_morris_cloud", "plot_all_morris_clouds",
    "plot_aggregate_morris_cloud", "plot_morris_cloud_interactive",
]


def morris_trajectories(k: int, r: int = 20, p: int = 8,
                        seed: int = 0) -> tuple:
    """r traiettorie di Morris nell'ipercubo unitario a k dimensioni.

    p è il numero di livelli della griglia, delta = p / (2*(p-1)) è
    la scelta classica che rende il disegno bilanciato (ogni fattore
    campiona i suoi livelli con la stessa frequenza).

    Ritorna (U, factor_of_step) dove:
      U               (r*(k+1), k), le traiettorie impilate
      factor_of_step  (r*(k+1),) int, quale fattore è stato mosso per
                      arrivare a quella riga, -1 per i punti di partenza
    """
    rng = np.random.default_rng(seed)
    delta = p / (2.0 * (p - 1.0))
    base_levels = np.arange(0, p // 2) / (p - 1.0)   # livelli ammessi per il punto base

    rows, moved = [], []
    for _ in range(r):
        x = rng.choice(base_levels, size=k)
        order = rng.permutation(k)
        signs = rng.choice([-1.0, 1.0], size=k)

        rows.append(x.copy())
        moved.append(-1)
        for i in order:
            step = signs[i] * delta
            if not (0.0 <= x[i] + step <= 1.0):
                step = -step                     # rimbalzo sul bordo
            x[i] = x[i] + step
            rows.append(x.copy())
            moved.append(int(i))

    return np.clip(np.array(rows), 0.0, 1.0), np.array(moved, dtype=int)


def elementary_effects(U: np.ndarray, factor_of_step: np.ndarray,
                       y: np.ndarray, k: int) -> np.ndarray:
    """Gli EE di un singolo output: matrice (r, k), NaN dove il passo
    non è calcolabile (Y non definita a un capo del passo)"""
    y = np.asarray(y, dtype=float)
    starts = np.flatnonzero(factor_of_step < 0)
    r = len(starts)
    out = np.full((r, k), np.nan)

    for t, s in enumerate(starts):
        end = starts[t + 1] if t + 1 < r else len(factor_of_step)
        for step in range(s + 1, end):
            i = factor_of_step[step]
            du = U[step, i] - U[step - 1, i]
            if du != 0:
                out[t, i] = (y[step] - y[step - 1]) / du
    return out


def morris_indices(U: np.ndarray, factor_of_step: np.ndarray,
                   Y: pd.DataFrame, factor_names: list) -> pd.DataFrame:
    """Tabella tidy (output, fattore, mu, mu_star, sigma, n_ee).

    n_ee è il numero di effetti elementari effettivamente calcolabili
    su r possibili: se è molto minore di r, quel fattore ha spesso
    portato Y fuori dal dominio di definizione (tipicamente oltre un
    limite di fattibilita') e le sue statistiche vanno prese con
    cautela, non lette come "poco influente"
    """
    k = len(factor_names)
    rows = []
    for col in Y.columns:
        ee = elementary_effects(U, factor_of_step, Y[col].to_numpy(), k)
        # un fattore senza nemmeno un EE calcolabile da una slice tutta
        # NaN: nanmean se ne lamenta, ma NaN è esattamente la risposta
        # giusta e la colonna n_ee lo rende visibile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(ee, axis=0)
            mu_star = np.nanmean(np.abs(ee), axis=0)
            sigma = np.nanstd(ee, axis=0, ddof=1)
        n_ee = np.sum(~np.isnan(ee), axis=0)
        for i, name in enumerate(factor_names):
            rows.append({"output": col, "fattore": name,
                         "mu": mu[i], "mu_star": mu_star[i],
                         "sigma": sigma[i], "n_ee": int(n_ee[i])})
    return pd.DataFrame(rows)


def ranking_table(df_morris: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Da tabella tidy a matrice fattori x output di mu*, ordinata per
    influenza media.

    normalize=True divide ogni colonna per il suo mu* massimo: gli
    output hanno unità diverse (MJ/pax/nmi, nmi, adimensionale) e
    senza normalizzare la media fra colonne non vuol dire niente. La
    colonna 'mu_star_medio' è quella su cui guardare il ranking
    complessivo, ma la decisione di cosa tenere si prende leggendo
    anche le singole colonne: un fattore che domina un solo output che
    ti interessa va tenuto anche se in media conta poco
    """
    wide = df_morris.pivot(index="fattore", columns="output", values="mu_star")
    if normalize:
        wide = wide / wide.max(axis=0)
    wide["mu_star_medio"] = wide.mean(axis=1)
    return wide.sort_values("mu_star_medio", ascending=False)


# ---------------------------------------------------------------------
# Le nubi: tabelle
# ---------------------------------------------------------------------

def morris_cloud_table(df_morris: pd.DataFrame, threshold: float = 0.1,
                       normalize: bool = True) -> pd.DataFrame:
    """Prepara la tabella dei punti della nube, un output alla volta.

    df_morris è la tabella tidy di morris_indices (output, fattore, mu,
    mu_star, sigma, n_ee).

    threshold è RELATIVO: un fattore è selezionato se la sua distanza
    dall'origine supera threshold * (distanza massima di quell'output).
    0.1 vuol dire "tengo tutto quello che arriva almeno al 10% del
    fattore più lontano di questo punto operativo". Non è che una soglia
    grafica, non un test statistico: serve a leggere il grafico, la
    decisione finale su cosa passa alla Sobol resta a mano.

    Colonne aggiunte:
      scala           il mu* massimo dell'output, cioè il divisore
      mu_star_n       mu* normalizzato (1 per il fattore dominante)
      sigma_n         sigma normalizzato con lo stesso divisore
      distanza        sqrt(mu_star_n^2 + sigma_n^2)
      soglia          la soglia assoluta usata per quell'output
      selezionato     distanza > soglia
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold è una frazione della distanza massima: "
                         "deve stare in [0, 1]")

    df = df_morris.copy()
    df["mu_star"] = df["mu_star"].astype(float)
    df["sigma"] = df["sigma"].astype(float)

    if normalize:
        scala = df.groupby("output")["mu_star"].transform("max")
    else:
        scala = pd.Series(1.0, index=df.index)
    # un output tutto NaN o tutto zero non si normalizza: si lascia com'è
    scala = scala.where(np.isfinite(scala) & (scala > 0), 1.0)

    df["scala"] = scala
    df["mu_star_n"] = df["mu_star"] / scala
    df["sigma_n"] = df["sigma"] / scala
    df["distanza"] = np.hypot(df["mu_star_n"], df["sigma_n"])
    df["soglia"] = threshold * df.groupby("output")["distanza"].transform("max")
    df["selezionato"] = df["distanza"] > df["soglia"]
    return df.sort_values(["output", "distanza"], ascending=[True, False]).reset_index(drop=True)


def aggregate_morris_cloud(df_morris: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """La nube cumulativa: media su tutti gli output delle coordinate
    normalizzate.

    Una riga per fattore:
      mu_star_medio     media di mu_star_n sugli output
      sigma_medio       media di sigma_n sugli output
      distanza          sqrt(mu_star_medio^2 + sigma_medio^2)
      mu_star_max       il massimo di mu_star_n: serve a non perdere un
                        fattore che domina un solo punto operativo e
                        sparisce nella media
      n_selezionato     in quanti output ha superato la soglia
      n_output          su quanti output è stato calcolato (esclusi
                        quelli dove mu* e' NaN, cioè nessun effetto
                        elementare calcolabile)
      selezionato       distanza > threshold * distanza massima

    La media è fatta ignorando i NaN: se un fattore è incalcolabile in
    un punto, quel punto non entra nella sua media, ma n_output lo
    registra
    """
    tab = morris_cloud_table(df_morris, threshold=threshold, normalize=True)
    valid = tab[np.isfinite(tab["mu_star_n"])]

    agg = (valid.groupby("fattore")
           .agg(mu_star_medio=("mu_star_n", "mean"),
                sigma_medio=("sigma_n", "mean"),
                mu_star_max=("mu_star_n", "max"),
                n_selezionato=("selezionato", "sum"),
                n_output=("mu_star_n", "size"),
                n_ee_min=("n_ee", "min"))
           .reset_index())
    agg["distanza"] = np.hypot(agg["mu_star_medio"], agg["sigma_medio"])
    agg["soglia"] = threshold * agg["distanza"].max()
    agg["selezionato"] = agg["distanza"] > agg["soglia"]
    return agg.sort_values("distanza", ascending=False).reset_index(drop=True)


def select_factors(df_morris: pd.DataFrame, threshold: float = 0.1,
                   dominanza: float = 0.5, use_aggregate: bool = True) -> list:
    """I nomi dei fattori da portare alla Sobol.

    use_aggregate=True (default): si tiene un fattore se

      - è selezionato sulla nube cumulativa (distanza media sopra
        soglia), OPPURE
      - in almeno un punto operativo il suo mu* normalizzato arriva a
        'dominanza' (default 0.5, cioè metà del fattore dominante di
        quel punto)

    La seconda condizione è lì apposta: un fattore che governa la
    batteria e non conta niente altrove ha una media bassa, perchè la
    batteria è 3 punti su 17, ma è esattamente il risultato che
    interessa. Contare in quanti punti supera la soglia non basterebbe:
    con una soglia al 10% quasi tutto la supera da qualche parte, ed è
    la ragione per cui il criterio guarda quanto è forte dove è
    forte, non in quanti posti è appena visibile.

    use_aggregate=False: unione dei selezionati output per output, il
    criterio più inclusivo di tutti.

    Resta una proposta da guardare, non una decisione automatica: la
    guideline chiede la scelta a mano, e sulla Sobol pesa direttamente
    il costo N*(k+2)
    """
    if not use_aggregate:
        tab = morris_cloud_table(df_morris, threshold=threshold)
        return sorted(tab.loc[tab["selezionato"], "fattore"].unique().tolist())

    agg = aggregate_morris_cloud(df_morris, threshold=threshold)
    keep = agg["selezionato"] | (agg["mu_star_max"] >= dominanza)
    return agg.loc[keep, "fattore"].tolist()


# ---------------------------------------------------------------------
# Le nubi: grafici
# ---------------------------------------------------------------------

def _annotate_no_overlap(ax, x, y, labels, fontsize=7.5, colors=None):
    """Etichette che non si sovrappongono, in modo elementare: per ogni
    punto si provano alcune posizioni attorno (destra/sinistra, via via
    più in alto o più in basso) e si prende la prima libera.

    I punti vengono etichettati in ordine di distanza decrescente
    dall'origine, così i fattori più influenti si prendono il posto
    migliore. Non è un algoritmo di label placement serio: nelle nubi
    molto affollate una manciata di etichette resta lontana dal suo
    punto, ma nessuna finisce sopra un'altra
    """
    candidates = [(7, 4, "left"), (7, -11, "left"), (-7, 4, "right"), (-7, -11, "right"),
                  (7, 17, "left"), (7, -24, "left"), (-7, 17, "right"), (-7, -24, "right"),
                  (7, 30, "left"), (-7, 30, "right"), (7, -37, "left"), (-7, -37, "right")]
    placed = []

    order = np.argsort(-np.hypot(np.asarray(x, float), np.asarray(y, float)))
    for idx in order:
        lab = str(labels[idx])
        px, py = ax.transData.transform((x[idx], y[idx]))
        w, h = 0.55 * fontsize * len(lab), 1.4 * fontsize

        chosen = candidates[0]
        for dx, dy, ha in candidates:
            x0 = px + dx if ha == "left" else px + dx - w
            box = (x0, py + dy - h / 2, x0 + w, py + dy + h / 2)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                chosen = (dx, dy, ha)
                placed.append(box)
                break
        else:
            dx, dy, ha = chosen
            x0 = px + dx if ha == "left" else px + dx - w
            placed.append((x0, py + dy - h / 2, x0 + w, py + dy + h / 2))

        dx, dy, ha = chosen
        # se l'etichetta è stata spostata lontano, un filo la ricollega
        # al suo punto: senza, in una nube affollata non si capisce più
        # di chi sia
        arrow = (dict(arrowstyle="-", lw=0.5, color="0.55", shrinkA=0, shrinkB=3)
                 if abs(dy) > 11 else None)
        ax.annotate(lab, (x[idx], y[idx]), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=fontsize, arrowprops=arrow,
                    color="black" if colors is None else colors[idx])


def _draw_cloud(ax, x, y, labels, selected, soglia, title, xlabel, ylabel,
                annotate_all: bool = False, fontsize: float = 7.5):
    """Il disegno vero e proprio, condiviso da nube singola e cumulativa"""
    sel = np.asarray(selected, dtype=bool)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ax.scatter(x[~sel], y[~sel], s=35, facecolors="none", edgecolors="0.6",
               linewidths=1.0, label=f"sotto soglia ({int((~sel).sum())})", zorder=2)
    ax.scatter(x[sel], y[sel], s=55, color="tab:red", zorder=3,
               label=f"selezionati ({int(sel.sum())})")

    lim = float(np.nanmax(np.hypot(x, y))) * 1.15 if len(x) else 1.0
    lim = max(lim, 1e-12)

    # arco della soglia: dentro l'arco = vicino all'origine = si butta
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(soglia * np.cos(theta), soglia * np.sin(theta), ls="--", lw=1.2,
            color="tab:red", alpha=0.7, label=f"soglia d = {soglia:.3g}")
    # riferimento sigma = mu*: sopra questa retta l'effetto dipende dal
    # punto più di quanto valga in media
    ax.plot([0, lim], [0, lim], ls=":", lw=1.0, color="0.4", alpha=0.8,
            label=r"$\sigma = \mu^*$")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    labels = list(labels)
    mask = np.ones(len(x), dtype=bool) if annotate_all else sel
    if mask.any():
        _annotate_no_overlap(ax, x[mask], y[mask],
                             [lab for lab, m in zip(labels, mask) if m],
                             fontsize=fontsize,
                             colors=["black" if s else "0.5"
                                     for s, m in zip(sel, mask) if m])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    return ax


def plot_morris_cloud(df_morris: pd.DataFrame, output: str, threshold: float = 0.1,
                      normalize: bool = True, ax=None, annotate_all: bool = False,
                      fontsize: float = 7.5):
    """La nube (mu*, sigma) di un punto operativo.

    output è il nome della colonna di Y, cioe' quello che compare nella
    colonna 'output' di df_morris (per esempio
    "E[Hydrogen combustion]@1500nmi_450kt_fan").

    normalize=False disegna mu* e sigma nelle unità vere
    [MJ/(pax*nmi)]: utile se il grafico va in tesi da solo e si vuole
    poter dire "questo fattore sposta l'intensity di tanto"
    """
    tab = morris_cloud_table(df_morris, threshold=threshold, normalize=normalize)
    sub = tab[tab["output"] == output]
    if sub.empty:
        raise ValueError(f"Nessun output {output!r} in df_morris. "
                         f"Disponibili: {sorted(df_morris['output'].unique())}")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.0, 6.0))

    unit = "" if normalize else " [MJ/(pax nmi)]"
    _draw_cloud(ax, sub["mu_star_n"], sub["sigma_n"], sub["fattore"],
                sub["selezionato"], float(sub["soglia"].iloc[0]),
                title=output,
                xlabel=r"$\mu^*$" + (" (normalizzato)" if normalize else unit),
                ylabel=r"$\sigma$" + (" (normalizzato)" if normalize else unit),
                annotate_all=annotate_all, fontsize=fontsize)
    if created:
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
    return ax


def plot_all_morris_clouds(df_morris: pd.DataFrame, threshold: float = 0.1,
                           ncols: int = 3, outputs: Optional[list] = None,
                           separate_figures: bool = False, figsize_per_ax=(4.2, 3.8)):
    """Una nube per ogni punto operativo.

    Di default le mette tutte in una griglia sola (con ~17 output sono
    17 riquadri: si guardano insieme, ed è così che si vede subito se
    il ranking cambia da un punto all'altro). separate_figures=True
    apre una figura per punto, formato tesi.

    outputs: sottoinsieme di nomi da disegnare, nell'ordine dato
    (comodo per raggruppare per sistema propulsivo)
    """
    names = outputs or list(pd.unique(df_morris["output"]))

    if separate_figures:
        axes = []
        for nm in names:
            ax = plot_morris_cloud(df_morris, nm, threshold=threshold)
            ax.legend(fontsize=8, loc="upper left")
            axes.append(ax)
        return axes

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_ax[0] * ncols,
                                                    figsize_per_ax[1] * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, nm in zip(axes, names):
        plot_morris_cloud(df_morris, nm, threshold=threshold, ax=ax, fontsize=6.5)
    for ax in axes[len(names):]:
        ax.axis("off")

    # legenda unica: quella dei singoli assi riporterebbe i conteggi del
    # primo riquadro, che non valgono per gli altri
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec="0.6", label="sotto soglia"),
        Line2D([], [], marker="o", ls="none", color="tab:red", label="selezionati"),
        Line2D([], [], ls="--", color="tab:red",
               label=f"soglia: d > {threshold:.0%} della distanza massima del riquadro"),
        Line2D([], [], ls=":", color="0.4", label=r"$\sigma = \mu^*$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Screening di Morris: nubi per punto operativo "
                 f"(soglia = {threshold:.0%} della distanza massima)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def plot_aggregate_morris_cloud(df_morris: pd.DataFrame, threshold: float = 0.1,
                                ax=None, annotate_all: bool = False,
                                show_counts: bool = True):
    """La nube cumulativa: un punto per fattore, media sulle nubi
    normalizzate di tutti i punti operativi.

    show_counts=True aggiunge all'etichetta "n=selezionati/totale", cioè
    in quanti punti operativi quel fattore aveva superato la soglia da
    solo: è l'informazione che la media da sola nasconde, ed è lo
    stesso indicatore dei tornado medi dell'analisi locale
    """
    agg = aggregate_morris_cloud(df_morris, threshold=threshold)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.5, 6.5))

    labels = agg["fattore"]
    if show_counts:
        labels = [f"{r.fattore}  (n={int(r.n_selezionato)}/{int(r.n_output)})"
                  for r in agg.itertuples()]

    _draw_cloud(ax, agg["mu_star_medio"], agg["sigma_medio"], labels,
                agg["selezionato"], float(agg["soglia"].iloc[0]),
                title="Nube cumulativa: media sui punti operativi "
                      f"(soglia = {threshold:.0%})",
                xlabel=r"$\overline{\mu^*}$ (normalizzato per output)",
                ylabel=r"$\overline{\sigma}$ (normalizzato per output)",
                annotate_all=annotate_all)
    if created:
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
    return ax


def plot_morris_cloud_interactive(df_morris: pd.DataFrame, output: Optional[str] = None,
                                  threshold: float = 0.1):
    """Nube con slider sulla soglia: si trascina e si vede quali fattori
    restano fuori dall'arco.

    output=None disegna la nube cumulativa. Serve un backend
    interattivo (in Jupyter: %matplotlib widget oppure %matplotlib
    qt), con il backend inline lo slider si disegna ma non risponde.

    Lo slider è per esplorare: la soglia scelta va poi riscritta come
    numero nello script, altrimenti il risultato non è riproducibile
    """
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    fig.subplots_adjust(bottom=0.18)
    slider_ax = fig.add_axes((0.15, 0.05, 0.7, 0.03))
    slider = Slider(slider_ax, "soglia (frazione della distanza max)",
                    0.0, 1.0, valinit=threshold, valstep=0.01)

    def redraw(val):
        ax.clear()
        if output is None:
            plot_aggregate_morris_cloud(df_morris, threshold=float(val), ax=ax)
        else:
            plot_morris_cloud(df_morris, output, threshold=float(val), ax=ax)
        ax.legend(fontsize=8, loc="upper left")
        fig.canvas.draw_idle()

    redraw(threshold)
    slider.on_changed(redraw)
    fig._morris_slider = slider      # senza un riferimento vivo lo slider muore
    return fig
