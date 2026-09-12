"""
Probabilistic technology map.

Prende il cubo grezzo prodotto in propagation
(cnav.uncertainty.propagation.PropagationResult) e ne ricava, per ogni
punto (R, V) della griglia, la probabilita' che ciascuna tecnologia sia
la migliore:

    P_j(R, V) = P[T(R, V) = j] ~ (1/N) * sum_k I[T^(k)(R, V) = j]

e la classificazione richiesta:

    P_j > 0.90  ->  regione ROBUSTA per la tecnologia j
    altrimenti  ->  DECISION-UNCERTAIN region

Qui dentro non si esegue mai il modello, tutto è post-processing del
cubo salvato. Cambiare la soglia di robustezza, aggregare fan ed elica,
o guardare un confine diverso non richiede di rilanciare la
propagazione.

Oltre alla mappa, il modulo produce due delle metriche finali:
  - intervalli di confidenza sulla renewable electricity
    intensity (intensity_percentiles)
  - posizione media e dispersione dei technology boundaries
    (boundary_statistics), che è il risultato su cui la guideline
    chiede di concentrare l'attenzione ("l'attenzione principale
    dovrebbe essere posta sui confini fra tecnologie").

UNA CAUTELA SULL'INTERPRETAZIONE
--------------------------------
P_j(R, V) è una probabilità rispetto alle incertezze considerate, non
una probabilità che quella tecnologia venga davvero adottata: dipende
interamente dalle PDF dichiarate e dalla regione accettabile individuata. 
Un P = 0.95 va letto come "la conclusione è insensibile
alle incertezze che ho messo dentro", non come una previsione. In
particolare, se un valore nominale sta sul bordo della sua PDF (vedi
distributions.check_nominal_consistency), la mappa probabilistica sarà
spostata rispetto a quella deterministica per costruzione, e la
differenza fra le due non è solo dispersione.
"""
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .propagation import PROPULSORS, TECHNOLOGIES, PropagationResult

__all__ = [
    "ProbabilityMap",
    "probability_map",
    "aggregate_to_technologies",
    "classify_regions",
    "intensity_percentiles",
    "feasibility_probability",
    "boundary_statistics",
    "adjacent_label_pairs",
    "BASE_COLORS",
    "label_color",
    "plot_probability_map",
    "plot_probability_field",
    "plot_intensity_band",
    "plot_boundary_dispersion",
]

BASE_COLORS = {
    "Battery-electric": "#4C72B0",
    "Hydrogen fuel cell": "#DD8452",
    "Hydrogen combustion": "#55A868",
    "e-SAF combustion": "#C44E52",
}


@dataclass
class ProbabilityMap:
    """P_j(R, V) per ogni etichetta j.

    probabilities: (n_labels, n_speeds, n_ranges), somma su asse 0 <= 1
    (può essere < 1 nei punti in cui, per qualche campione, nessuna
    configurazione e' fattibile).
    """
    probabilities: np.ndarray
    labels: list
    ranges_nmi: np.ndarray
    speeds_kt: np.ndarray
    n_samples: int

    @property
    def dominant_index(self) -> np.ndarray:
        """Indice della tecnologia più probabile in ogni punto"""
        return np.argmax(self.probabilities, axis=0)

    @property
    def dominant_probability(self) -> np.ndarray:
        """P della tecnologia più probabile, cioè max_j P_j(R, V)"""
        return np.max(self.probabilities, axis=0)

    def probability_of(self, label: str) -> np.ndarray:
        return self.probabilities[self.labels.index(label)]

    def to_dataframe(self) -> pd.DataFrame:
        """Formato lungo (una riga per punto di griglia e tecnologia),
        comodo per esportare i numeri della mappa in tesi"""
        rows = []
        for j, label in enumerate(self.labels):
            for i, v in enumerate(self.speeds_kt):
                for k, r in enumerate(self.ranges_nmi):
                    rows.append({"range_nmi": r, "speed_kt": v,
                                 "tecnologia": label,
                                 "P": self.probabilities[j, i, k]})
        return pd.DataFrame(rows)


def probability_map(result: PropagationResult) -> ProbabilityMap:
    """Conta, punto per punto, quante volte ciascuna etichetta vince.

    I campioni in cui nessuna configurazione è fattibile (best_index =
    -1) non vengono conteggiati per nessuna tecnologia: in quei punti la
    somma delle P è minore di 1, e la differenza è la probabilità che
    il punto sia irraggiungibile. Non si normalizza a 1 di nascosto,
    perchè quell'informazione è il limite di fattibilità
    probabilistico ed è uno degli output di riferimento chiesti
    """
    n_labels = len(result.labels)
    counts = np.zeros((n_labels,) + result.best_index.shape[1:], dtype=np.int32)
    for j in range(n_labels):
        counts[j] = np.sum(result.best_index == j, axis=0)

    return ProbabilityMap(
        probabilities=counts / result.n_samples,
        labels=list(result.labels),
        ranges_nmi=result.grid.ranges_nmi,
        speeds_kt=result.grid.speeds_kt,
        n_samples=result.n_samples,
    )


def aggregate_to_technologies(pmap: ProbabilityMap) -> ProbabilityMap:
    """Somma fan ed elica dentro la stessa tecnologia: da 8 etichette a 4.

    Utile per la domanda "quale VETTORE ENERGETICO conviene", che è la
    domanda della tesi; la mappa a 8 etichette risponde invece alla
    domanda più fine "quale vettore e quale architettura propulsiva",
    che è quella della mappa deterministica dell'articolo.

    Le due mappe possono dare risposte diverse nello stesso punto: se in
    un punto la batteria vince con l'elica nel 45% dei campioni e con il
    fan nel 20%, nessuna delle 8 etichette è robusta ma la tecnologia
    "batteria" lo è al 65%. Conviene mostrarle entrambe e commentare la
    differenza, che è esattamente incertezza sulla scelta
    dell'architettura a parità di vettore energetico
    """
    agg = np.zeros((len(TECHNOLOGIES),) + pmap.probabilities.shape[1:])
    for j, label in enumerate(pmap.labels):
        tech_name = label.rsplit(" (", 1)[0]
        agg[TECHNOLOGIES.index(tech_name)] += pmap.probabilities[j]
    return ProbabilityMap(
        probabilities=agg, labels=list(TECHNOLOGIES),
        ranges_nmi=pmap.ranges_nmi, speeds_kt=pmap.speeds_kt,
        n_samples=pmap.n_samples,
    )


def classify_regions(pmap: ProbabilityMap, threshold: float = 0.90) -> np.ndarray:
    """Classificazione robusta / decision-uncertain.

    Ritorna un array (n_speeds, n_ranges) di interi: l'indice della
    tecnologia dove max_j P_j > threshold, e -1 dove nessuna tecnologia
    raggiunge la soglia (decision-uncertain region).

    La soglia 0.90 è quella suggerita dalla guideline; conviene
    riportare in tesi anche la sensibilità del risultato alla soglia
    (es. area delle regioni robuste per threshold = 0.80, 0.90, 0.95),
    perchè la superficie delle regioni robuste dipende molto da questo
    numero e sceglierne uno solo nasconde quella dipendenza
    """
    out = pmap.dominant_index.copy()
    out[pmap.dominant_probability <= threshold] = -1
    return out


def robust_area_fraction(pmap: ProbabilityMap,
                         thresholds: Sequence[float] = (0.80, 0.90, 0.95)) -> pd.DataFrame:
    """Frazione di punti di griglia classificati come robusti, al variare
    della soglia, è la diagnostica suggerita sopra: una riga per soglia"""
    rows = []
    for t in thresholds:
        classified = classify_regions(pmap, threshold=t)
        rows.append({"soglia": t,
                     "frazione_robusta": float(np.mean(classified >= 0)),
                     "frazione_incerta": float(np.mean(classified < 0))})
    return pd.DataFrame(rows)


def intensity_percentiles(result: PropagationResult, label: str,
                          percentiles: Sequence[float] = (5, 50, 95)) -> dict:
    """Intervalli di confidenza sull'electricity intensity.

    Ritorna {percentile: array (n_speeds, n_ranges)}. I NaN (campioni in
    cui quella configurazione non è fattibile) sono esclusi dal calcolo
    dei percentili: il percentile è quindi condizionato alla
    fattibilità. Questo va detto quando si riporta una banda di
    confidenza per la batteria ai raggi lunghi, dove la banda è
    calcolata su una frazione sempre più piccola di campioni - usare
    feasibility_probability() per sapere quanto piccola
    """
    j = result.labels.index(label)
    data = result.intensities[:, j, :, :]
    # nei punti in cui la configurazione non e' MAI fattibile la colonna e'
    # tutta NaN e numpy avvisa: e' il comportamento voluto (il percentile
    # resta NaN), quindi l'avviso si silenzia invece di sporcare l'output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {p: np.nanpercentile(data, p, axis=0) for p in percentiles}


def feasibility_probability(result: PropagationResult, label: str) -> np.ndarray:
    """P(configurazione fattibile) in ogni punto (R, V).

    Questa è la versione probabilistica del "limite di fattibilitaà delle
    configurazioni battery-electric" elencato fra gli output di
    riferimento: invece di una singola curva di range
    massimo si ottiene una transizione graduale da 1 a 0, la cui
    larghezza misura quanto quel limite sia incerto
    """
    j = result.labels.index(label)
    return np.mean(~np.isnan(result.intensities[:, j, :, :]), axis=0)


def boundary_statistics(result: PropagationResult, label_a: str, label_b: str,
                        percentiles: Sequence[float] = (5, 50, 95)) -> pd.DataFrame:
    """Posizione e dispersione del confine fra due tecnologie.

    Per ogni velocità di crociera e per ogni campione k, si cerca il
    range in cui la differenza di intensity fra label_a e label_b cambia
    segno, cioè il punto in cui le due tecnologie si equivalgono. La
    ricerca è fatta per interpolazione lineare in log(R), coerente con
    la griglia geometrica, e quindi non è limitata alla risoluzione
    della griglia stessa.

    Ritorna un DataFrame con una riga per velocità: media, deviazione
    standard e percentili della posizione del confine, più la frazione
    di campioni in cui il confine esiste davvero dentro la griglia. Se
    quest'ultima è bassa il confine è fuori dal dominio esplorato per
    la maggior parte dei campioni, e media e dispersione delle poche
    righe rimaste sono poco significative: è una colonna da guardare
    sempre prima delle altre
    """
    ja, jb = result.labels.index(label_a), result.labels.index(label_b)
    log_r = np.log(result.grid.ranges_nmi)
    rows = []

    for i, speed in enumerate(result.grid.speeds_kt):
        crossings = []
        for k in range(result.n_samples):
            diff = result.intensities[k, ja, i, :] - result.intensities[k, jb, i, :]
            valid = ~np.isnan(diff)
            if valid.sum() < 2:
                continue
            d, lr = diff[valid], log_r[valid]
            sign_change = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)
            if sign_change.size == 0:
                continue
            m = sign_change[0]   # il primo attraversamento a partire dal raggio corto
            t = d[m] / (d[m] - d[m + 1])
            crossings.append(float(np.exp(lr[m] + t * (lr[m + 1] - lr[m]))))

        row = {"speed_kt": float(speed),
               "frazione_con_confine": len(crossings) / result.n_samples}
        if crossings:
            arr = np.array(crossings)
            row["media_nmi"] = float(arr.mean())
            row["std_nmi"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            for p in percentiles:
                row[f"p{int(p)}_nmi"] = float(np.percentile(arr, p))
        rows.append(row)

    return pd.DataFrame(rows)


def adjacent_label_pairs(pmap: "ProbabilityMap", min_cells: int = 3) -> list:
    """Le coppie di etichette i cui domini si toccano sulla mappa.

    Va usata per scegliere su quali coppie ha senso chiamare
    boundary_statistics. Quest'ultima infatti cerca l'incrocio fra due
    curve di intensity senza verificare che una delle due sia
    effettivamente il minimo su tutte le etichette: per una coppia che
    sulla mappa non si tocca mai, l'incrocio esiste ed è calcolabile,
    ma è sepolto sotto una terza tecnologia e non è un confine.
    Con 8 etichette le coppie possibili sono 28 e quelle vere sono
    tipicamente 3-5.

    Scorre la mappa del vincitore (argmax_j P_j) e conta, per ogni
    coppia, quante volte le due etichette compaiono in celle contigue
    lungo il range o lungo la velocità. min_cells scarta i contatti di
    una o due celle, che di norma sono l'effetto della risoluzione
    finita della griglia attorno a un punto triplo e non un confine
    esteso.

    Ritorna [(label_a, label_b, n_celle_di_contatto), ...] ordinata per
    lunghezza di contatto decrescente.
    """
    dom = pmap.dominant_index
    counts = {}
    for left, right in ((dom[:, :-1], dom[:, 1:]), (dom[:-1, :], dom[1:, :])):
        differing = left != right
        for a, b in zip(left[differing], right[differing]):
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0) + 1

    out = [(pmap.labels[a], pmap.labels[b], n)
           for (a, b), n in counts.items() if n >= min_cells and a >= 0 and b >= 0]
    return sorted(out, key=lambda t: -t[2])


# =====================================================================
# Grafici
# =====================================================================

def _label_color(label: str) -> str:
    """Colore della tecnologia, schiarito per l'elica: stessa convenzione
    dei grafici deterministici (execution/01_riproduzione/plot_best_system_map*.py), così
    la mappa probabilistica è confrontabile a colpo d'occhio con quella
    deterministica"""
    name = label.rsplit(" (", 1)[0] if " (" in label else label
    base = BASE_COLORS[name]
    if "propeller)" not in label:
        return base
    h = base.lstrip("#")
    r, g, b = (min(255, int(int(h[i:i + 2], 16) * 1.35)) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"


# alias pubblico: serve agli script che disegnano mappe a 8 etichette con
# la stessa convenzione di colore della mappa probabilistica
label_color = _label_color


def plot_probability_map(pmap: ProbabilityMap, threshold: float = 0.90,
                         ax=None, title: Optional[str] = None):
    """La mappa probabilistica: colore = tecnologia più probabile,
    intensità del colore = quanto è probabile.

    Le regioni decision-uncertain (max_j P_j <= threshold) restano
    visibili ma slavate, invece di essere cancellate: è il modo più
    diretto di mostrare che la mappa deterministica traccia una linea
    netta dove in realtà c'è una fascia di indecisione
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))

    dominant = pmap.dominant_index
    p_max = pmap.dominant_probability

    # RGB del dominante, sbiancato in proporzione all'incertezza:
    # alpha va da 0.15 quando P e' al minimo possibile (1/n_labels) a 1
    # quando P = 1.
    p_floor = 1.0 / len(pmap.labels)
    alpha = np.clip((p_max - p_floor) / (1.0 - p_floor), 0.0, 1.0)
    alpha = 0.15 + 0.85 * alpha

    rgb = np.zeros(dominant.shape + (3,))
    for j, label in enumerate(pmap.labels):
        mask = dominant == j
        rgb[mask] = to_rgb(_label_color(label))
    image = 1.0 - alpha[..., None] * (1.0 - rgb)   # blend verso il bianco

    ax.pcolormesh(pmap.ranges_nmi, pmap.speeds_kt, image, shading="auto")

    # contorno della soglia di robustezza
    ax.contour(pmap.ranges_nmi, pmap.speeds_kt, p_max,
               levels=[threshold], colors="k", linewidths=1.2, linestyles="--")

    ax.set_xscale("log")
    ax.set_xlabel("Range [nmi]")
    ax.set_ylabel("Velocità di crociera [kt]")
    ax.set_title(title or f"Tecnologia più probabile (N = {pmap.n_samples})")

    present = sorted(set(dominant.flatten()))
    handles = [Patch(color=_label_color(pmap.labels[j]), label=pmap.labels[j])
               for j in present]
    handles.append(Patch(facecolor="white", edgecolor="k", linestyle="--",
                         label=f"confine P = {threshold:.2f}"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)
    return ax


def plot_probability_field(pmap: ProbabilityMap, label: str, ax=None,
                           levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)):
    """P_j(R, V) per una tecnologia, come campo continuo.

    Complementare alla mappa "chi vince": mostra dove una tecnologia è
    competitiva anche quando non è la più probabile, informazione che
    la mappa a colori per definizione nasconde
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6))

    p = pmap.probability_of(label)
    mesh = ax.pcolormesh(pmap.ranges_nmi, pmap.speeds_kt, p,
                         cmap="viridis", vmin=0, vmax=1, shading="auto")
    contours = ax.contour(pmap.ranges_nmi, pmap.speeds_kt, p, levels=list(levels),
                          colors="w", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.set_xscale("log")
    ax.set_xlabel("Range [nmi]")
    ax.set_ylabel("Velocità di crociera [kt]")
    ax.set_title(f"P[{label} è la migliore]")
    plt.colorbar(mesh, ax=ax, label="probabilità")
    return ax


def plot_intensity_band(result: PropagationResult, speed_kt: float,
                        labels: Optional[Sequence[str]] = None,
                        percentiles: Sequence[float] = (5, 50, 95), ax=None):
    """Bande di confidenza dell'electricity intensity vs range, a velocità
    fissata: la versione probabilistica della Fig. 4 dell'articolo.

    speed_kt viene approssimata al nodo di griglia più vicino (nessuna
    interpolazione: si mostrano i dati calcolati, non una loro
    ricostruzione)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    i = int(np.argmin(np.abs(result.grid.speeds_kt - speed_kt)))
    actual_speed = float(result.grid.speeds_kt[i])
    labels = list(labels) if labels is not None else list(result.labels)
    lo, mid, hi = sorted(percentiles)

    for label in labels:
        pct = intensity_percentiles(result, label, percentiles=(lo, mid, hi))
        color = _label_color(label)
        ax.plot(result.grid.ranges_nmi, pct[mid][i], color=color, lw=1.8, label=label)
        ax.fill_between(result.grid.ranges_nmi, pct[lo][i], pct[hi][i],
                        color=color, alpha=0.18, lw=0)

    ax.set_xscale("log")
    ax.set_xlabel("Range [nmi]")
    ax.set_ylabel("Electricity intensity [MJ/(pax*nmi)]")
    ax.set_title(f"Intensity a {actual_speed:.0f} kt - mediana e banda "
                 f"p{int(lo)}-p{int(hi)} su {result.n_samples} campioni")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_boundary_dispersion(stats_df: pd.DataFrame, ax=None,
                             title: Optional[str] = None,
                             min_fraction: float = 0.5):
    """Posizione e dispersione di un confine fra tecnologie vs velocità.

    Le velocità in cui il confine esiste in meno di min_fraction dei
    campioni vengono scartate: la loro media sarebbe calcolata su una
    minoranza non rappresentativa (vedi boundary_statistics)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))

    df = stats_df[stats_df["frazione_con_confine"] >= min_fraction].dropna(subset=["media_nmi"])
    if df.empty:
        raise ValueError(
            f"Nessuna velocità ha un confine in almeno il {100*min_fraction:.0f}% "
            "dei campioni: le due tecnologie non si incrociano nel dominio esplorato")

    ax.fill_betweenx(df["speed_kt"], df["p5_nmi"], df["p95_nmi"],
                     alpha=0.25, color="#4C72B0", lw=0, label="p5-p95")
    ax.plot(df["p50_nmi"], df["speed_kt"], color="#4C72B0", lw=2, label="mediana")

    ax.set_xscale("log")
    ax.set_xlabel("Range del confine [nmi]")
    ax.set_ylabel("Velocità di crociera [kt]")
    ax.set_title(title or "Posizione del confine fra tecnologie")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
