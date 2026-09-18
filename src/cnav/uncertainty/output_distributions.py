"""
Distribuzione dell'output in un punto operativo.

La mappa probabilistica (technology_map.py) risponde alla domanda "chi
vince dove", e le bande di confidenza (plot_intensity_band) mostrano
mediana e p5-p95 lungo tutto l'asse dei range. Nessuna delle due mostra
la cosa più elementare che la propagazione ha prodotto: in un singolo
punto (R, V), e per un singolo sistema propulsivo, l'insieme dei valori
di intensity ottenuti dagli N campioni di theta è un campione della
PDF dell'output, e quella PDF si può disegnare.

Sarebbe l'output della Fase 7 nella sua forma più diretta: una banda p5-p95
riassume la distribuzione in due numeri e nasconde se sia simmetrica,
asimmetrica o bimodale; qui la si guarda intera.

DUE SORGENTI PER GLI STESSI CAMPIONI
------------------------------------
    samples_from_result   legge il cubo già salvato
                          (propagation_result.npz), costo zero, ma il
                          punto richiesto viene approssimato al nodo di
                          griglia più vicino
    samples_by_rerun      rivaluta il modello nel punto esatto,
                          riusando la theta salvata dentro il risultato

La seconda serve quando il punto operativo è uno di quelli usati
altrove nella tesi (MISSION_LIBRARY dell'analisi di sensibilità: 100,
1500, 6000 nmi) e si vuole che sia lo stesso punto e non il nodo
vicino: la griglia di default è geometrica con passo del 17% in range,
quindi lo scarto dello snap arriva all'8%. Costa una rivalutazione del
modello per campione e per sistema (~0.7 ms), cioè pochi secondi per
punto con N = 1500, contro le ore della propagazione completa.

UNA CAUTELA, LA STESSA DI intensity_percentiles
-----------------------------------------------
I NaN sono configurazioni non fattibili per quel theta, non valori
mancanti. La PDF disegnata è quindi condizionata alla fattibilità, e
va letta insieme a P(fattibile): una batteria al limite della sua
autonomia produce una distribuzione stretta e innocua calcolata sul 10%
dei campioni. Per questo il default di plot_intensity_pdf scala ogni
densità per P(fattibile) - l'area sotto la curva diventa la
probabilità che quella configurazione esista, non 1, e P(fatt) finisce
comunque in legenda
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .propagation import PROPULSORS, TECHNOLOGIES, PropagationResult, theta_row_to_tech_wtt
from .technology_map import label_color

__all__ = [
    "OperatingPoint",
    "PointSample",
    "labels_for_propulsor",
    "samples_from_result",
    "samples_by_rerun",
    "nominal_intensities",
    "plot_intensity_pdf",
    "plot_difference_pdf",
    "plot_points_comparison",
]


@dataclass(frozen=True)
class OperatingPoint:
    """Un punto operativo: range, velocità di crociera, propulsore.

    Il propulsore fa parte del punto e non del sistema: a 450 kt l'elica
    non è fattibile per nessuna tecnologia, quindi mettere fan ed elica
    nello stesso grafico produce metà figura vuota. name è solo
    l'etichetta per titoli e nomi di file (comodo riusare le chiavi di
    MISSION_LIBRARY: "medium_fan", "long_propeller", ...).
    """
    range_nmi: float
    speed_kt: float
    propulsor: str = "fan"
    name: str = ""

    def __post_init__(self):
        if self.propulsor not in PROPULSORS:
            raise ValueError(f"propulsor deve essere uno di {PROPULSORS}, "
                             f"non {self.propulsor!r}")

    @property
    def title(self) -> str:
        base = f"{self.range_nmi:g} nmi, {self.speed_kt:g} kt, {self.propulsor}"
        return f"{self.name} ({base})" if self.name else base

    @property
    def slug(self) -> str:
        return (self.name or
                f"{self.range_nmi:g}nmi_{self.speed_kt:g}kt_{self.propulsor}")


def labels_for_propulsor(propulsor: str) -> list:
    """Le 4 etichette tecnologia x propulsore per un propulsore, nello
    stesso ordine di technology_labels()"""
    return [f"{tech} ({propulsor})" for tech in TECHNOLOGIES]


@dataclass
class PointSample:
    """I campioni dell'output in un punto, una colonna per sistema.

    values: {etichetta: array (N,) float, con NaN dove non fattibile}.
    I NaN sono tenuti apposta: sono l'informazione sulla fattibilità e
    servono ad allineare le colonne (la riga k è lo stesso theta in
    tutte le colonne, ed è quello che rende sensato il confronto a
    coppie di plot_difference_pdf).

    range_nmi / speed_kt sono il punto effetivamente valutato: con
    samples_from_result è il nodo di griglia, che può differire da
    quello richiesto
    """
    point: OperatingPoint
    values: dict
    range_nmi: float
    speed_kt: float
    source: str = "cubo"

    @property
    def labels(self) -> list:
        return list(self.values)

    @property
    def n_samples(self) -> int:
        return len(next(iter(self.values.values())))

    @property
    def snap_error(self) -> tuple:
        """(scarto relativo in range, scarto assoluto in velocità) fra
        il punto richiesto e quello valutato"""
        return (abs(self.range_nmi - self.point.range_nmi) / self.point.range_nmi,
                abs(self.speed_kt - self.point.speed_kt))

    def feasible(self, label: str) -> np.ndarray:
        """I soli valori fattibili, senza NaN"""
        v = self.values[label]
        return v[~np.isnan(v)]

    def p_feasible(self, label: str) -> float:
        return float(np.mean(~np.isnan(self.values[label])))

    def p_best(self, label: str) -> float:
        """P[questo sistema è il migliore] FRA QUELLI PRESENTI in values.

        Non è la P della mappa probabilistica, che confronta tutte e 8
        le etichette: qui il confronto è ristretto ai sistemi che
        stanno nel grafico (di norma le 4 tecnologie con lo stesso
        propulsore). Le due cose coincidono solo dove l'altro
        propulsore non è comunque competitivo.

        Il conto è fatto riga per riga, cioè a theta fissato: è
        questo che lo rende diverso dal guardare la sovrapposizione
        delle PDF marginali, che ignora la correlazione fra sistemi
        """
        stack = np.vstack([self.values[l] for l in self.labels])
        all_nan = np.all(np.isnan(stack), axis=0)
        safe = np.where(np.isnan(stack), np.inf, stack)
        winner = np.argmin(safe, axis=0)
        return float(np.mean((winner == self.labels.index(label)) & ~all_nan))

    def summary(self, percentiles: Sequence[float] = (5, 50, 95)) -> pd.DataFrame:
        """Tabella riassuntiva: una riga per sistema.

        cv è il coefficiente di variazione sd/media, l'indicatore
        adimensionale con cui confrontare la dispersione di sistemi che
        stanno su ordini di grandezza diversi
        """
        rows = []
        for label in self.labels:
            v = self.feasible(label)
            row = {"sistema": label,
                   "P_fattibile": self.p_feasible(label),
                   "n_fattibili": int(v.size),
                   "P_migliore": self.p_best(label)}
            if v.size:
                row.update({"media": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan})
                row["cv"] = row["sd"] / row["media"] if row["media"] else np.nan
                for p in percentiles:
                    row[f"p{p:g}"] = float(np.percentile(v, p))
            rows.append(row)
        return pd.DataFrame(rows)


def samples_from_result(result: PropagationResult, point: OperatingPoint,
                        labels: Optional[Sequence[str]] = None,
                        tol_range: float = 0.05) -> PointSample:
    """I campioni di output nel nodo di griglia più vicino al punto.

    Non interpola: si mostrano i valori calcolati. Se il nodo dista più
    di tol_range (relativo) dal range richiesto, la cosa viene stampata
    invece di passare inosservata - è il caso in cui conviene
    samples_by_rerun.
    """
    labels = list(labels) if labels is not None else labels_for_propulsor(point.propulsor)

    i = int(np.argmin(np.abs(result.grid.speeds_kt - point.speed_kt)))
    j = int(np.argmin(np.abs(np.log(result.grid.ranges_nmi) - np.log(point.range_nmi))))
    r_act, v_act = float(result.grid.ranges_nmi[j]), float(result.grid.speeds_kt[i])

    err_r = abs(r_act - point.range_nmi) / point.range_nmi
    if err_r > tol_range:
        print(f"  nota: {point.range_nmi:g} nmi -> nodo {r_act:.0f} nmi "
              f"({100 * err_r:.1f}% di scarto). Per il punto esatto usa "
              f"samples_by_rerun()")

    values = {l: np.asarray(result.intensities[:, result.labels.index(l), i, j],
                            dtype=float) for l in labels}
    return PointSample(point=point, values=values, range_nmi=r_act,
                       speed_kt=v_act, source="cubo")


def samples_by_rerun(theta: pd.DataFrame, point: OperatingPoint,
                     labels: Optional[Sequence[str]] = None,
                     form=None, verbose: bool = False) -> PointSample:
    """Rivaluta il modello nel punto esatto, su ogni riga di theta.

    theta è la stessa matrice della propagazione (result.theta la
    riporta: è salvata dentro il .npz), quindi i campioni ottenuti qui
    sono gli stessi della mappa probabilistica, solo valutati in un
    punto che non è un nodo di griglia. Costo: N x n_sistemi
    valutazioni del modello, dell'ordine di 0.7 ms ciascuna
    """
    # import locali: questo modulo e' anche solo post-processing di un
    # .npz, e in quel caso non deve tirarsi dietro tutto il modello
    from ..model.aircraft_sizer import AircraftSizer
    from ..model.energy_intensity import compute_intensity
    from ..model.mission import Mission
    from ..model.propulsion_systems import build_default_systems
    from ..model.well_to_tank import build_energy_carriers

    labels = list(labels) if labels is not None else labels_for_propulsor(point.propulsor)
    tech_names = [l.rsplit(" (", 1)[0] for l in labels]
    mission = Mission(range_nmi=float(point.range_nmi),
                      cruise_speed_kt=float(point.speed_kt),
                      propulsor=point.propulsor)

    out = {l: np.full(len(theta), np.nan) for l in labels}
    for k in range(len(theta)):
        tech, wtt = theta_row_to_tech_wtt(theta.iloc[k].to_dict(), form)
        sizer = AircraftSizer(tech)
        systems = {s.name: s for s in build_default_systems(build_energy_carriers(wtt))}
        for label, tech_name in zip(labels, tech_names):
            try:
                res = compute_intensity(mission, systems[tech_name], tech, sizer)
                out[label][k] = res.intensity_MJ_per_pax_nmi
            except Exception:
                pass          # resta NaN: non fattibile per questo theta
        if verbose and (k + 1) % 250 == 0:
            print(f"    {k + 1}/{len(theta)} campioni")

    return PointSample(point=point, values=out, range_nmi=float(point.range_nmi),
                       speed_kt=float(point.speed_kt), source="rivalutato")


def nominal_intensities(point: OperatingPoint,
                        labels: Optional[Sequence[str]] = None,
                        form=None, overrides: Optional[dict] = None) -> dict:
    """Il valore del modello a theta fissato, nello stesso punto.

    Serve a sovrapporre alla PDF una riga verticale di riferimento. Con
    overrides=None si usano i valori di default di TechAssumptions e
    WellToTankEfficiencies, cioè il modello dopo la calibrazione:
    è il riferimento giusto per dire quanto ci si è spostati
    dall'articolo
    """
    riga = pd.DataFrame([dict(overrides)]) if overrides else pd.DataFrame(index=[0])
    fissato = samples_by_rerun(riga, point, labels=labels, form=form)
    return {l: float(v[0]) for l, v in fissato.values.items()}


# =====================================================================
# Grafici
# =====================================================================

def _density(values: np.ndarray, grid: np.ndarray):
    """KDE gaussiana, None se i campioni non bastano o sono degeneri"""
    from scipy.stats import gaussian_kde
    if values.size < 5 or np.ptp(values) <= 0:
        return None
    try:
        return gaussian_kde(values)(grid)
    except Exception:
        return None


def plot_intensity_pdf(sample: PointSample, ax=None,
                       labels: Optional[Sequence[str]] = None,
                       bins: int = 40, kde: bool = True,
                       scale_by_feasibility: bool = True,
                       log_x: bool = False,
                       show_median: bool = True,
                       nominal: Optional[dict] = None,
                       xlim: Optional[tuple] = None):
    """Le PDF dell'output nel punto, una curva per sistema.

    Istogramma normalizzato a densità + KDE sovrapposta. Con
    scale_by_feasibility (default) ogni curva è moltiplicata per
    P(fattibile), così l'area sotto la curva è la probabilità che
    quella configurazione esista e un sistema fattibile in un terzo dei
    campioni non sembra alla pari con uno fattibile sempre.

    log_x serve quando i sistemi mostrati stanno su ordini di grandezza
    diversi (tipico ai bordi del dominio, dove una configurazione
    diverge vicino al proprio limite): su scala lineare la curva stretta
    diventa una riga verticale e quella larga schiaccia tutto. Occhio
    pero': su asse logaritmico l'area non è più leggibile come
    probabilità.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    labels = list(labels) if labels is not None else sample.labels
    vivi = [l for l in labels if sample.feasible(l).size >= 2]
    if not vivi:
        raise ValueError(f"Nessun sistema fattibile in {sample.point.title}: "
                         "non c'è nessuna distribuzione da disegnare")

    # Con log_x la densità è stimata sul LOGARITMO dei campioni, non sui
    # campioni: una KDE fatta in lineare e poi disegnata su asse
    # logaritmico schiaccia contro l'asse proprio le distribuzioni larghe,
    # che sono quelle per cui serviva il log. Qui l'ordinata è quindi una
    # densità per decade, e l'area sotto la curva resta leggibile
    trasf = np.log10 if log_x else (lambda x: x)

    tutti = np.concatenate([sample.feasible(l) for l in vivi])
    lo, hi = (float(np.min(tutti)), float(np.max(tutti))) if xlim is None else xlim
    t_lo, t_hi = trasf(max(lo, 1e-9)) if log_x else lo, trasf(hi) if log_x else hi
    margine = 0.05 * (t_hi - t_lo) if t_hi > t_lo else 1.0
    t_lo, t_hi = t_lo - margine, t_hi + margine

    t_griglia = np.linspace(t_lo, t_hi, 400)
    t_bordi = np.linspace(t_lo, t_hi, bins + 1)
    griglia = 10.0 ** t_griglia if log_x else t_griglia
    bordi = 10.0 ** t_bordi if log_x else t_bordi

    for label in vivi:
        v = sample.feasible(label)
        colore = label_color(label)
        peso = sample.p_feasible(label) if scale_by_feasibility else 1.0

        conteggi, _ = np.histogram(trasf(v), bins=t_bordi, density=True)
        ax.stairs(conteggi * peso, bordi, fill=True, alpha=0.18,
                  color=colore, lw=0)

        etichetta = (f"{label.rsplit(' (', 1)[0]}  "
                     f"[P(fatt)={sample.p_feasible(label):.2f}, "
                     f"P(mig)={sample.p_best(label):.2f}]")
        dens = _density(trasf(v), t_griglia) if kde else None
        if dens is not None:
            ax.plot(griglia, dens * peso, color=colore, lw=1.8, label=etichetta)
        else:
            ax.stairs(conteggi * peso, bordi, color=colore, lw=1.8, label=etichetta)

        if show_median:
            ax.axvline(np.median(v), color=colore, ls=":", lw=1.1, alpha=0.9)
        if nominal and label in nominal and np.isfinite(nominal[label]):
            ax.axvline(nominal[label], color=colore, ls="--", lw=1.3, alpha=0.9)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(griglia[0], griglia[-1])
    ax.set_xlabel("Electricity intensity [MJ/(pax*nmi)]")
    ax.set_ylabel(("densità per decade" if log_x else "densità di probabilità") +
                  (" x P(fattibile)" if scale_by_feasibility else ""))
    ax.set_title(f"Distribuzione dell'output - {sample.point.title}\n"
                 f"N = {sample.n_samples} campioni, punto {sample.source} "
                 f"a {sample.range_nmi:.0f} nmi / {sample.speed_kt:.0f} kt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    note = []
    if show_median:
        note.append("punteggiata: mediana")
    if nominal:
        note.append("tratteggiata: valore deterministico")
    if note:
        ax.text(0.99, 0.02, " | ".join(note), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, style="italic", color="0.35")
    return ax


def plot_difference_pdf(sample: PointSample, label_a: str, label_b: str,
                        ax=None, bins: int = 40):
    """La PDF della DIFFERENZA E_a - E_b nello stesso punto.

    Complemento necessario del grafico precedente. Due PDF
    marginali molto sovrapposte non implicano che il confronto sia
    incerto: i due sistemi sono valutati sullo stesso theta, e se
    condividono i parametri dominanti si spostano insieme, per cui la
    differenza può avere segno praticamente certo mentre le marginali
    si accavallano. Solo la distribuzione della differenza dice se il
    confronto in quel punto è deciso o no.

    La massa a sinistra dello zero è P[a meglio di b], calcolata sui
    soli campioni in cui entrambi sono fattibili
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    d = sample.values[label_a] - sample.values[label_b]
    d = d[~np.isnan(d)]
    if d.size < 2:
        raise ValueError(f"{label_a} e {label_b} non sono mai fattibili insieme "
                         f"in {sample.point.title}")

    p_a = float(np.mean(d < 0))
    frazione_coppie = d.size / sample.n_samples

    bordi = np.linspace(float(d.min()), float(d.max()), bins + 1)
    conteggi, _ = np.histogram(d, bins=bordi, density=True)
    ax.stairs(conteggi, bordi, fill=True, alpha=0.30, color="#4C72B0", lw=0)
    griglia = np.linspace(bordi[0], bordi[-1], 400)
    dens = _density(d, griglia)
    if dens is not None:
        ax.plot(griglia, dens, color="#4C72B0", lw=1.8)

    ax.axvline(0.0, color="k", lw=1.4)
    ax.set_xlabel(f"E[{label_a}] - E[{label_b}]  [MJ/(pax*nmi)]")
    ax.set_ylabel("densità di probabilita'")
    ax.set_title(f"Confronto a theta fissato - {sample.point.title}\n"
                 f"P[{label_a.rsplit(' (', 1)[0]} migliore] = {p_a:.3f} "
                 f"(su {100 * frazione_coppie:.0f}% dei campioni, "
                 f"quelli con entrambi fattibili)")
    ax.grid(alpha=0.3)
    return ax


def plot_points_comparison(samples: Sequence[PointSample], label: str, ax=None,
                           percentiles: Sequence[float] = (5, 50, 95),
                           clip_percentiles: Optional[tuple] = None):
    """Lo stesso sistema in più punti operativi, come violini affiancati.

    La figura d'insieme da mettere in tesi prima dei singoli
    ingrandimenti: mostra come la distribuzione si sposta e si allarga
    passando da corto a lungo raggio, e dove la fattibilità comincia a
    perdere campioni (la percentuale sotto ogni violino)

    clip_percentiles = (lo, hi) calcola la sagoma del violino sui soli
    campioni fra quei due percentili, punto per punto. Serve dove una
    manciata di campioni esplode (vicino a un limite di fattibilità, o
    oltre il picco della curva dell'elica): con le code dentro, la KDE
    allarga la banda e schiaccia tutti i violini in schegge piatte.
    Ritagliare il solo asse con set_ylim non basta, perchè la sagoma è
    già stata calcolata sull'intero campione.

    I campioni esclusi restano dati veri, non outlier da scartare.
    Mediana e banda restano calcolate sul campione COMPLETO (sono
    percentili, le code non li spostano) e la figura riporta quanti
    campioni sono finiti fuori dall'asse: la nota va tenuta in
    didascalia, altrimenti il grafico mostra una dispersione minore di
    quella che la propagazione ha prodotto
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    dati, completi, posizioni, nomi, quote = [], [], [], [], []
    n_esclusi = 0
    for i, s in enumerate(samples):
        v = s.feasible(label)
        if v.size < 2:
            continue
        v_plot = v
        if clip_percentiles is not None:
            c_lo, c_hi = np.percentile(v, sorted(clip_percentiles))
            v_plot = v[(v >= c_lo) & (v <= c_hi)]
            n_esclusi += int(v.size - v_plot.size)
            if v_plot.size < 2:      # taglio troppo aggressivo: meglio tutto
                v_plot = v
        dati.append(v_plot)
        completi.append(v)
        posizioni.append(i)
        nomi.append(s.point.name or f"{s.point.range_nmi:g} nmi")
        quote.append(s.p_feasible(label))

    if not dati:
        raise ValueError(f"{label} non è fattibile in nessuno dei punti dati")

    parti = ax.violinplot(dati, positions=posizioni, showextrema=False, widths=0.75)
    for corpo in parti["bodies"]:
        corpo.set_facecolor(label_color(label))
        corpo.set_alpha(0.35)

    # mediana e banda sul campione completo, non su quello tagliato
    lo, mid, hi = sorted(percentiles)
    for pos, v in zip(posizioni, completi):
        ax.plot([pos, pos], [np.percentile(v, lo), np.percentile(v, hi)],
                color=label_color(label), lw=1.6)
        ax.plot(pos, np.percentile(v, mid), "o", color=label_color(label), ms=5)

    for pos, q in zip(posizioni, quote):
        ax.annotate(f"P(fatt)\n{q:.2f}", (pos, 0), xytext=(0, -28),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="0.35",
                    xycoords=("data", "axes fraction"))

    ax.set_xticks(posizioni)
    ax.set_xticklabels(nomi, fontsize=8)
    ax.set_ylabel("Electricity intensity [MJ/(pax*nmi)]")
    ax.set_title(f"{label} nei punti operativi "
                 f"(mediana e banda p{lo:g}-p{hi:g})")
    ax.grid(alpha=0.3, axis="y")

    if clip_percentiles is not None:
        c_lo, c_hi = sorted(clip_percentiles)
        basso = min(float(v.min()) for v in dati)
        alto = max(float(v.max()) for v in dati)
        margine = 0.08 * (alto - basso) if alto > basso else 1.0
        ax.set_ylim(basso - margine, alto + margine)
        ax.text(0.99, 0.98, f"sagoma sui campioni fra p{c_lo:g} e p{c_hi:g}; "
                            f"{n_esclusi} campioni fuori dall'asse",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                style="italic", color="0.35")
    return ax