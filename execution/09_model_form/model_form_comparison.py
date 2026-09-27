"""
Confronto fra le forme del modello (model-form uncertainty).

Esegue la propagazione dell'incertezza sotto ciascuna variante
concettuale e confronta le mappe probabilistiche risultanti con quella
del modello base.

Uso:
    python execution/09_model_form/model_form_comparison.py

Presuppone theta_acc.csv. Il costo è N_SAMPLES * (numero di forme)
propagazioni complete: partire con N piccolo e griglia rada per vedere
il flusso, e solo dopo lanciare il run da tesi.

Cosa produce:
  model_form_<slug>.npz         il cubo grezzo di ciascuna forma
  model_form_deterministic.png  le mappe deterministiche affiancate
  model_form_maps.png           le mappe probabilistiche affiancate
  model_form_disagreement.png   dove le mappe non sono d'accordo
  model_form_maps_<slug>.png, model_form_deterministic_<slug>.png
                                una figura per forma (FIGURE_SINGOLE)
  model_form_disagreement_<slug>.png
                                il disaccordo di una variante per volta
                                (FIGURE_SINGOLE)
  model_form_maps_griglia.png, model_form_deterministic_griglia.png
                                le forme di FORME_GRIGLIA in griglia

La tabella dei parametri per forma e quella degli spostamenti vengono
stampate a schermo e basta: sono da leggere mentre gira, non da
archiviare, e il dato definitivo sta nei .npz

Come si legge il risultato
--------------------------
La domanda non è "quale forma è giusta": nessuna lo è, sono tutte
approssimazioni difendibili. La domanda è quanto le conclusioni della
tesi dipendono dalla scelta. Tre livelli di risposta, in ordine di
gravità crescente:

  1. le mappe si assomigliano, i confini si spostano di poco: la
     model-form uncertainty è piccola rispetto a quella parametrica, e
     le conclusioni reggono
  2. i confini si spostano ma l'ordine delle tecnologie no: si riporta
     lo spostamento come banda di incertezza aggiuntiva sui confini
  3. cambia quale tecnologia vince in una regione: lì la conclusione
     dipende da un'ipotesi di modellazione, non dai dati, e va detto
     esplicitamente. Questo è il tipo di risultato più scomodo e 
     anche il più interessante da scrivere

La colonna 'frazione_celle_cambiate' della tabella degli spostamenti è
la misura diretta del livello 3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnav.model.constants import TechAssumptions, WellToTankEfficiencies
from cnav.model.energy_intensity import compute_intensity
from cnav.model.mission import Mission
from cnav.model.model_form import MODEL_FORM_PRESETS
from cnav.model.propulsion_systems import build_default_systems
from cnav.model.well_to_tank import build_energy_carriers
from cnav.uncertainty import load_theta_acc
from cnav.uncertainty.model_form_uncertainty import (
    assemble_theta_for_form,
    parameter_membership_table,
    specs_for_form,
)
from cnav.uncertainty.propagation import (
    PROPULSORS,
    FlightGrid,
    run_propagation,
    technology_labels,
)
from cnav.uncertainty.technology_map import (
    TECHNOLOGIES,
    aggregate_to_technologies,
    label_color,
    plot_probability_map,
    probability_map,
)

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
THETA_ACC_PATH = Path(__file__).resolve().parents[1] / "06_incertezza_calibrazione" / "theta_acc.csv"

# le forme da confrontare. Ognuna isola una variante, poi "tutte" le
# somma: confrontare solo base contro tutte direbbe CHE qualcosa cambia
# ma non QUALE ipotesi lo sta causando
FORMS = ["base", "raymer", "polare", "easa", "tutte"]

N_SAMPLES = 750            # 60 per provare, 500-1000 per il run da tesi
N_RANGES = 45             # griglia rada per la prova, 45 x 40 per la tesi
N_SPEEDS = 40
SEED = 0
N_WORKERS = 1             # > 1 richiede il blocco if __name__ (c'è già)

# --- checkpoint -------------------------------------------------------
# USA_CHECKPOINT = False: niente cartelle _ckpt_*, niente ripresa
# RESUME = False: le cartelle vengono scritte ma non rilette
# In ogni caso, cambiando N_SAMPLES o la griglia serve una cartella
# nuova: i campioni LHS non sono gli stessi (vedi _run_marker)
USA_CHECKPOINT = True
RESUME = True

# SALVA_CUBI: i .npz per forma. Sono l'unico artefatto autosufficiente
# (contengono anche theta e la griglia, che i checkpoint non hanno) e
# servono per rifare percentili e statistiche dei confini senza
# rilanciare la propagazione
SALVA_CUBI = True

# Risoluzione delle sole FIGURE:
#   False -> fan ed elica distinti, 8 etichette
#   True  -> aggregate alle 4 tecnologie
# Gli indici di confronto restano comunque a 8 etichette
MAPPE_AGGREGATE = False

# Figure aggiuntive, oltre a quelle con tutte le forme affiancate
#   FIGURE_SINGOLE  una figura per forma:
#                   model_form_maps_<slug>.png, model_form_deterministic_<slug>.png
#                   e, per le sole varianti, model_form_disagreement_<slug>.png
#   FORME_GRIGLIA   le forme della figura a griglia, riempita per righe:
#                   model_form_maps_griglia.png e model_form_deterministic_griglia.png
#                   Lista vuota = niente griglia
FIGURE_SINGOLE = True
FORME_GRIGLIA = ["base", "raymer", "polare", "easa"]
GRIGLIA_NCOLS = 2

# dimensione (pollici) di tutte le figure singole: la stessa di
# best_system_map.png e di probabilistic_technology_map.png, per cui sono
# pensate le legende dentro il pannello
FIGSIZE_SINGOLA = (9.5, 6.5)

# dimensione del titolo del pannello (non del suptitle): nelle figure
# singole il pannello è più grande e il titolo cresce con lui
FONTSIZE_TITOLO_PANNELLO = 12
FONTSIZE_TITOLO_SINGOLA = 15

# Titoli dei pannelli, uno per forma. None = titolo automatico, cioè
# quello di sempre:
#   mappe deterministiche e probabilistiche  "<nome>: <label del preset>"
#                                            (solo "base" per il modello base)
#   mappe di disaccordo                      "<nome>: xx.x% delle celle"
# Con una stringa, quella stringa prende il posto del titolo in tutte le
# figure; nelle mappe di disaccordo le si aggiunge ": xx.x% delle celle".
# Esempio:  "raymer": "Stima dei pesi secondo Raymer",
TITOLI_FORME = {
    "base":   "Baseline Model",
    "raymer": "Raymer OEW Fraction Estimation",
    "polare": "Altitude-Sensitive Aerodynamic Efficiency",
    "easa":   "EASA Conform Reserve Formulation",
    "tutte":  "ALL",
}

LABELS = technology_labels()

OUT_DIR = Path(__file__).resolve().parent


def confronta_mappe(pmap_base, pmap_var) -> dict:
    """Le tre misure di distanza fra due mappe probabilistiche.

    frazione_celle_cambiate  in quante celle cambia la tecnologia più
        probabile, è la misura che conta per le conclusioni: se è zero
        la variante sposta le probabilità ma non le decisioni
    distanza_media_L1  la metà della distanza L1 fra i vettori di
        probabilità, mediata sulle celle. Sta in [0, 1] e vale 0 per
        mappe identiche e 1 per mappe che non hanno alcuna
        sovrapposizione. Cattura anche gli spostamenti che non
        ribaltano il vincitore
    p_max_media_delta  quanto cambia in media la confidenza nel
        vincitore. Positivo = la variante rende le mappe più nette
    """
    P0, P1 = pmap_base.probabilities, pmap_var.probabilities
    vincitore0, vincitore1 = np.argmax(P0, axis=0), np.argmax(P1, axis=0)
    return {
        "frazione_celle_cambiate": float(np.mean(vincitore0 != vincitore1)),
        "distanza_media_L1": float(np.mean(0.5 * np.sum(np.abs(P0 - P1), axis=0))),
        "p_max_media_delta": float(np.mean(np.max(P1, axis=0) - np.max(P0, axis=0))),
    }


def aggrega_indici(mappa: np.ndarray) -> np.ndarray:
    """Da indici sulle 8 etichette a indici sulle 4 tecnologie.

    L'equivalente di technology_map.aggregate_to_technologies per una
    mappa di soli vincitori invece che di probabilita'. Il -1 (nessun
    sistema fattibile) resta -1
    """
    out = np.full_like(mappa, -1)
    for k, label in enumerate(LABELS):
        out[mappa == k] = TECHNOLOGIES.index(label.rsplit(" (", 1)[0])
    return out


def mappa_deterministica(grid: FlightGrid, form) -> np.ndarray:
    """La mappa del sistema migliore con i parametri al valore nominale.

    È la mappa "classica" di execution/01_riproduzione/plot_best_system_map.py, ricalcolata
    sotto una data forma del modello: nessun campionamento, un solo
    velivolo per punto di griglia, la tecnologia con la minor electricity
    intensity. Costa un campione invece di N, quindi è praticamente
    gratis rispetto alla propagazione.

    Serve a due cose. La prima è di controllo: la mappa probabilistica
    dovrebbe assomigliarle, e dove non le assomiglia c'è qualcosa da
    capire (di norma è una regione dove il vincitore nominale vince di
    poco, e basta poca incertezza per ribaltarlo). La seconda è di
    lettura: separa l'effetto della forma del modello, che si vede già
    qui, dall'effetto dell'incertezza dei parametri, che è la differenza
    fra questa figura e quella probabilistica.

    Ritorna un array (n_speeds, n_ranges) di indici in LABELS, con -1
    dove nessun sistema è fattibile. La distinzione fan/elica è
    mantenuta: un passaggio da fan a elica a parità di vettore
    energetico è un cambiamento vero, e aggregarlo prima di contare le
    celle lo cancella dagli indici. L'aggregazione alle 4 tecnologie
    resta disponibile a valle con aggrega_indici(), per le sole figure
    """

    tech = TechAssumptions(model_form=form)
    carriers = build_energy_carriers(WellToTankEfficiencies())
    systems = build_default_systems(carriers)

    n_speeds, n_ranges = grid.shape
    best = np.full((n_speeds, n_ranges), -1, dtype=int)

    for i, speed_kt in enumerate(grid.speeds_kt):
        for j, range_nmi in enumerate(grid.ranges_nmi):
            migliore, valore_migliore = -1, np.inf
            for p_idx, propulsor in enumerate(PROPULSORS):
                mission = Mission(range_nmi=float(range_nmi),
                                  cruise_speed_kt=float(speed_kt),
                                  propulsor=propulsor)
                for s_idx, system in enumerate(systems):
                    try:
                        v = compute_intensity(mission, system, tech).intensity_MJ_per_pax_nmi
                    except Exception:
                        continue
                    # il confronto con se stesso scarta i NaN senza
                    # bisogno di isnan: NaN < x è sempre falso
                    if v == v and v < valore_migliore:
                        valore_migliore = v
                        # stessa convenzione di propagation.evaluate_sample,
                        # cioe' l'ordine di technology_labels()
                        migliore = s_idx * len(PROPULSORS) + p_idx
            best[i, j] = migliore
    return best


def titolo_forma(nome: str, breve: bool = False) -> str:
    """Il titolo del pannello di una forma, vedi TITOLI_FORME.

    breve=True è la versione delle mappe di disaccordo, dove il titolo
    automatico è il solo nome della forma
    """
    personalizzato = TITOLI_FORME.get(nome)
    if personalizzato is not None:
        return personalizzato
    forma = MODEL_FORM_PRESETS[nome]
    if breve or forma.is_baseline:
        return nome
    return f"{nome}: {forma.label}"


def _layout(n: int, ncols) -> tuple:
    """(righe, colonne) per n pannelli; ncols=None = tutti su una riga"""
    ncols = n if ncols is None else min(ncols, n)
    return -(-n // ncols), ncols


def plot_mappe_deterministiche(mappe: dict, grid: FlightGrid, etichette: list,
                               ncols=None,
                               titolo="Best Carbon-Neutral System under different model assumption\n"
                                      "(Deterministic Map)"):
    """Le mappe nominali delle forme in `mappe`, su una griglia di pannelli.

    Colori uguali a quelli della mappa probabilistica (label_color), così
    le due figure si leggono una sotto l'altra senza dover reimparare la
    legenda. Il grigio è "nessun sistema fattibile". ncols=None mette
    tutti i pannelli su una riga; con una sola forma esce la figura singola,
    e lì la legenda va dentro il pannello, semitrasparente, invece che sotto
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    nomi = list(mappe)
    colori = ["#BBBBBB"] + [label_color(e) for e in etichette]
    cmap = ListedColormap(colori)

    presenti = sorted({int(v) for m in mappe.values() for v in np.unique(m)})
    handles = [Patch(color=colori[k + 1],
                     label=etichette[k] if k >= 0 else "nessuno fattibile")
               for k in presenti]
    nrows, ncols = _layout(len(nomi), ncols)
    interna = len(nomi) == 1        # figura singola: legenda sopra la mappa
    ncol_leg = max(1, min(len(handles), 4 if ncols >= 4 else ncols))
    h_leg = 0.0 if interna else 0.3 + 0.36 * (-(-len(handles) // ncol_leg))  # pollici
    h_tit = 0.75
    larghezza = max(4.8 * ncols, 6.5)
    altezza = 3.8 * nrows + h_leg + h_tit
    if interna:
        # come la figura di plot_best_system_map: la legenda a 15 punti
        # dentro il pannello è pensata per questa dimensione
        larghezza, altezza = FIGSIZE_SINGOLA

    fig, axes = plt.subplots(nrows, ncols, figsize=(larghezza, altezza),
                             squeeze=False)
    for ax in axes.flat[len(nomi):]:
        ax.set_visible(False)
    for ax, nome in zip(axes.flat, nomi):
        # +1 perche' -1 (non fattibile) deve finire sul primo colore
        ax.pcolormesh(grid.ranges_nmi, grid.speeds_kt, mappe[nome] + 1,
                      cmap=cmap, vmin=-0.5, vmax=len(colori) - 0.5,
                      shading="auto")
        ax.set_xscale("log")
        ax.set_xlabel("Range [nmi]")
        ax.set_ylabel("Cruise Speed [kt]")
        ax.set_title(titolo_forma(nome), fontweight='bold',
                     fontsize=FONTSIZE_TITOLO_SINGOLA if interna
                     else FONTSIZE_TITOLO_PANNELLO)

    if interna:
        # stesse impostazioni della legenda di plot_probability_map
        axes.flat[0].legend(handles=handles, loc="upper right", fontsize=15,
                            framealpha=0.2)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=ncol_leg, fontsize=14)
    fig.suptitle(titolo, fontsize=16, fontweight='bold')
    # il suptitle lo conta già tight_layout: il rect riserva solo la legenda
    fig.tight_layout(rect=(0, h_leg / altezza, 1, 1))
    return fig


def plot_mappe_probabilistiche(pmaps: dict, nomi: list, ncols=None,
                               titolo="Best Carbon-Neutral System under different model assumption\n"
                                      "(Probabilistic Map)"):
    """Le mappe probabilistiche delle forme in `nomi`, su una griglia di
    pannelli. ncols=None = tutte su una riga; una sola forma = figura
    singola.

    plot_probability_map mette una legenda dentro ogni pannello. Nella
    figura singola resta quella; con più pannelli la si toglie e se ne fa
    una sola, in basso, come per le deterministiche. Le
    voci vengono prese dalle legende dei pannelli prima di rimuoverle, così
    restano identiche (colori e confine P = soglia compresi) senza dover
    sapere come le costruisce technology_map
    """
    nrows, ncols = _layout(len(nomi), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(nomi):]:
        ax.set_visible(False)

    comune = len(nomi) > 1          # figura singola: legenda di default
    voci = {}                                   # testo -> handle, senza doppioni
    for ax, nome in zip(axes.flat, nomi):
        pm = aggregate_to_technologies(pmaps[nome]) if MAPPE_AGGREGATE else pmaps[nome]
        plot_probability_map(pm, ax=ax)
        ax.set_xlabel("Range [nmi]")
        ax.set_ylabel("Cruise Speed [kt]")
        ax.set_title(titolo_forma(nome), fontweight='bold',
                     fontsize=FONTSIZE_TITOLO_PANNELLO if comune
                     else FONTSIZE_TITOLO_SINGOLA)

        leg = ax.get_legend()
        if comune and leg is not None:
            # legend_handles da matplotlib 3.7, legendHandles prima
            handles = getattr(leg, "legend_handles", None) or leg.legendHandles
            for h, t in zip(handles, leg.get_texts()):
                voci.setdefault(t.get_text(), h)
            leg.remove()

    # prima le tecnologie nell'ordine di sempre, poi il resto (il confine)
    ordine = list(TECHNOLOGIES) if MAPPE_AGGREGATE else LABELS
    testi = sorted(voci, key=lambda t: ordine.index(t) if t in ordine else len(ordine))

    ncol_leg = max(1, min(len(testi), 4 if ncols >= 4 else ncols))
    h_leg = (0.3 + 0.36 * (-(-len(testi) // ncol_leg))       # pollici, per fontsize 14
             if testi else 0.0)
    h_tit = 0.75
    altezza = 4.4 * nrows + h_leg + h_tit
    larghezza = 5.2 * ncols
    if not comune:
        # come la mappa di plot_probabilistic_technology_map, per cui è
        # dimensionata la legenda di default di plot_probability_map
        larghezza, altezza = FIGSIZE_SINGOLA
    fig.set_size_inches(larghezza, altezza)

    if testi:
        fig.legend([voci[t] for t in testi], testi, loc="lower center",
                   ncol=ncol_leg, fontsize=14)
    fig.suptitle(titolo, fontsize=16, fontweight='bold')
    # il suptitle lo conta già tight_layout: il rect riserva solo la legenda
    fig.tight_layout(rect=(0, h_leg / altezza, 1, 1))
    return fig


def plot_disaccordo(pmaps: dict, varianti: list, ncols=None,
                    titolo="Where the model's variation changes the Best System\n"
                           "(Red = different from Baseline)"):
    """Per ogni variante, le celle in cui la tecnologia più probabile è
    diversa da quella del modello base: una cella accesa = lì la variante
    cambia la decisione. ncols=None = tutte su una riga; una sola variante
    = figura singola
    """
    nrows, ncols = _layout(len(varianti), ncols)
    dimensione = (FIGSIZE_SINGOLA if len(varianti) == 1
                  else (max(4.6 * ncols, 6.0), 4.0 * nrows + 0.5))
    # constrained: lascia a matplotlib lo spazio per il suptitle, che
    # nella figura singola va su più righe
    fig, axes = plt.subplots(nrows, ncols, figsize=dimensione,
                             squeeze=False, layout="constrained")
    for ax in axes.flat[len(varianti):]:
        ax.set_visible(False)
    vincitore_base = np.argmax(pmaps["base"].probabilities, axis=0)
    for ax, nome in zip(axes.flat, varianti):
        diverso = (np.argmax(pmaps[nome].probabilities, axis=0) != vincitore_base)
        ax.pcolormesh(pmaps["base"].ranges_nmi, pmaps["base"].speeds_kt,
                      diverso.astype(float), cmap="Reds", vmin=0, vmax=1,
                      shading="auto")
        ax.set_xscale("log")
        ax.set_xlabel("Range [nmi]")
        ax.set_ylabel("Cruise Speed [kt]")
        # nella singola il titolo entra in una riga; affiancate, a capo
        singola = len(varianti) == 1
        sep = " " if singola else "\n"
        ax.set_title(f"{titolo_forma(nome, breve=True)}:{sep}"
                     f"{diverso.mean():.1%} of the cells", fontweight='bold',
                     fontsize=FONTSIZE_TITOLO_SINGOLA if singola
                     else FONTSIZE_TITOLO_PANNELLO)
    fig.suptitle(titolo, fontsize=16, fontweight='bold')
    return fig


def controlla_forme_griglia():
    mancanti = [nm for nm in FORME_GRIGLIA if nm not in FORMS]
    if mancanti:
        raise ValueError(f"FORME_GRIGLIA contiene forme non in FORMS: {mancanti}")
    ignoti = [nm for nm in TITOLI_FORME if nm not in MODEL_FORM_PRESETS]
    if ignoti:
        raise ValueError(f"TITOLI_FORME contiene forme inesistenti: {ignoti}")


def confronta_mappe_deterministiche(mappe: dict, riferimento: str = "base") -> pd.DataFrame:
    """Quanto si sposta la mappa nominale rispetto a quella base.

    Da leggere accanto alla stessa misura sulle mappe probabilistiche:
    se una variante sposta molto la mappa deterministica ma poco quella
    probabilistica, significa che l'incertezza dei parametri copriva già
    quello spostamento, cioè che la scelta di modello non aggiunge
    granché a quanto già non si sapeva. Il caso opposto, poco qui e
    molto là, indica una variante che cambia soprattutto la larghezza
    delle zone di indecisione più che la loro posizione
    """
    base = mappe[riferimento]
    righe = []
    for nome, m in mappe.items():
        if nome == riferimento:
            continue
        righe.append({
            "forma": nome,
            "etichetta": MODEL_FORM_PRESETS[nome].label,
            "frazione_celle_cambiate_det": float(np.mean(m != base)),
        })
    return pd.DataFrame(righe)


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)

    # prima di tutto: scoprirlo dopo ore di propagazione sarebbe uno spreco
    controlla_forme_griglia()

    theta_acc = load_theta_acc(THETA_ACC_PATH)
    grid = FlightGrid.default(n_ranges=N_RANGES, n_speeds=N_SPEEDS)

    # -----------------------------------------------------------------
    # Chi esiste sotto quale forma
    # -----------------------------------------------------------------
    membership = parameter_membership_table(
        {nm: MODEL_FORM_PRESETS[nm] for nm in FORMS})

    print("=" * 78)
    print("PARAMETRI INCERTI PER FORMA DEL MODELLO")
    print("=" * 78)
    print(membership.to_string(index=False))
    print("\n  I parametri 'comuni' hanno le stesse PDF sotto ogni forma: è ciò")
    print("  che rende attribuibile alla forma, e non all'incertezza parametrica,")
    print("  la differenza fra due mappe")

    # -----------------------------------------------------------------
    # Mappe deterministiche (parametri al nominale): un campione a forma
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MAPPE DETERMINISTICHE")
    print("=" * 78)
    mappe_det = {nome: mappa_deterministica(grid, MODEL_FORM_PRESETS[nome])
                 for nome in FORMS}
    det_shift = confronta_mappe_deterministiche(mappe_det)
    print(det_shift.round(4).to_string(index=False))

    etichette_fig = list(TECHNOLOGIES) if MAPPE_AGGREGATE else LABELS
    mappe_fig = ({n: aggrega_indici(m) for n, m in mappe_det.items()}
                 if MAPPE_AGGREGATE else mappe_det)
    fig = plot_mappe_deterministiche(mappe_fig, grid, etichette_fig)
    det_path = OUT_DIR / "model_form_deterministic.png"
    fig.savefig(det_path, dpi=140)
    plt.close(fig)
    figure_extra = []

    if FIGURE_SINGOLE:
        for nome in FORMS:
            fig = plot_mappe_deterministiche(
                {nome: mappe_fig[nome]}, grid, etichette_fig,
                titolo="Deterministic Map")
            p = OUT_DIR / f"model_form_deterministic_{MODEL_FORM_PRESETS[nome].slug}.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            figure_extra.append(p)

    if FORME_GRIGLIA:
        fig = plot_mappe_deterministiche(
            {nm: mappe_fig[nm] for nm in FORME_GRIGLIA}, grid, etichette_fig,
            ncols=GRIGLIA_NCOLS)
        p = OUT_DIR / "model_form_deterministic_griglia.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figure_extra.append(p)

    # -----------------------------------------------------------------
    # Propagazione, una per forma
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PROPAGAZIONE")
    print("=" * 78)
    print(f"{N_SAMPLES} campioni x griglia {N_SPEEDS} x {N_RANGES} x {len(FORMS)} forme")

    pmaps = {}
    for nome in FORMS:
        form = MODEL_FORM_PRESETS[nome]
        print(f"\n--- {nome} ({form.label}) ---")
        print(form.describe())

        theta = assemble_theta_for_form(N_SAMPLES, theta_acc, form, seed=SEED)
        print(f"  {len(specs_for_form(form))} parametri di letteratura, "
              f"theta {theta.shape}")

        # ogni forma vuole la sua cartella di checkpoint: la firma scritta
        # da _run_marker fa fallire il tentativo di riusarne una per una
        # forma, una N o una griglia diverse
        ckpt = (str(OUT_DIR / f"_ckpt_model_form_{form.slug}")
                if USA_CHECKPOINT else None)
        result = run_propagation(
            theta, grid=grid, n_workers=N_WORKERS, verbose=True,
            checkpoint_dir=ckpt, resume=RESUME, form=form,
        )
        if SALVA_CUBI:
            result.save(str(OUT_DIR / f"model_form_{form.slug}.npz"))

        # gli indici si calcolano sempre sulle 8 etichette; l'aggregazione
        # e' solo una scelta di leggibilita' delle figure
        pmaps[nome] = probability_map(result)
        print(f"  celle senza alcuna tecnologia fattibile: "
              f"{result.infeasible_fraction:.1%}")

    # -----------------------------------------------------------------
    # Confronto
    # -----------------------------------------------------------------
    righe = []
    for nome in FORMS:
        if nome == "base":
            continue
        riga = {"forma": nome, "etichetta": MODEL_FORM_PRESETS[nome].label}
        riga.update(confronta_mappe(pmaps["base"], pmaps[nome]))
        righe.append(riga)
    shift = pd.DataFrame(righe).merge(
        det_shift.drop(columns="etichetta"), on="forma", how="left")

    print("\n" + "=" * 78)
    print("SPOSTAMENTO DELLA MAPPA RISPETTO AL MODELLO BASE")
    print("=" * 78)
    print(shift.round(4).to_string(index=False))
    print("\n  frazione_celle_cambiate > 0 significa che in quelle celle la")
    print("  tecnologia consigliata dipende da un'ipotesi di modellazione e non")
    print("  dai dati: è il risultato da dichiarare, non da nascondere")
    print("\n  Le due ultime colonne vanno confrontate fra loro: _det è lo")
    print("  spostamento della mappa nominale, l'altra quello della mappa")
    print("  probabilistica. Molto _det e poco l'altra = l'incertezza dei")
    print("  parametri copriva già quello spostamento")

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------
    fig = plot_mappe_probabilistiche(pmaps, FORMS)
    maps_path = OUT_DIR / "model_form_maps.png"
    fig.savefig(maps_path, dpi=140)
    plt.close(fig)

    if FIGURE_SINGOLE:
        for nome in FORMS:
            fig = plot_mappe_probabilistiche(pmaps, [nome],
                                             titolo="Probabilistic Map")
            p = OUT_DIR / f"model_form_maps_{MODEL_FORM_PRESETS[nome].slug}.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            figure_extra.append(p)

    if FORME_GRIGLIA:
        fig = plot_mappe_probabilistiche(pmaps, FORME_GRIGLIA,
                                         ncols=GRIGLIA_NCOLS)
        p = OUT_DIR / "model_form_maps_griglia.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figure_extra.append(p)

    # dove le mappe non sono d'accordo: una cella accesa = lì la
    # variante cambia la tecnologia consigliata
    varianti = [nm for nm in FORMS if nm != "base"]
    fig = plot_disaccordo(pmaps, varianti)
    dis_path = OUT_DIR / "model_form_disagreement.png"
    fig.savefig(dis_path, dpi=140)
    plt.close(fig)

    if FIGURE_SINGOLE:
        for nome in varianti:
            fig = plot_disaccordo(
                pmaps, [nome],
                titolo="Where the model's variation\nchanges the Best System\n"
                       "(Red = different from Baseline)")
            p = OUT_DIR / f"model_form_disagreement_{MODEL_FORM_PRESETS[nome].slug}.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            figure_extra.append(p)

    print(f"\nFigure salvate in {OUT_DIR}:")
    for p in (det_path, maps_path, dis_path, *figure_extra):
        print(f"  {p.name}")

    plt.close("all")


if __name__ == "__main__":
    main()