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

N_SAMPLES = 60            # 60 per provare, 500-1000 per il run da tesi
N_RANGES = 20             # griglia rada per la prova, 45 x 40 per la tesi
N_SPEEDS = 16
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


def plot_mappe_deterministiche(mappe: dict, grid: FlightGrid, etichette: list):
    """Le mappe nominali di tutte le forme, affiancate.

    Colori uguali a quelli della mappa probabilistica (label_color), così
    le due figure si leggono una sotto l'altra senza dover reimparare la
    legenda. Il grigio è "nessun sistema fattibile"
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    nomi = list(mappe)
    colori = ["#BBBBBB"] + [label_color(e) for e in etichette]
    cmap = ListedColormap(colori)

    fig, axes = plt.subplots(1, len(nomi), figsize=(4.8 * len(nomi), 4.3),
                             squeeze=False)
    for ax, nome in zip(axes[0], nomi):
        # +1 perche' -1 (non fattibile) deve finire sul primo colore
        ax.pcolormesh(grid.ranges_nmi, grid.speeds_kt, mappe[nome] + 1,
                      cmap=cmap, vmin=-0.5, vmax=len(colori) - 0.5,
                      shading="auto")
        ax.set_xscale("log")
        ax.set_xlabel("Range [nmi]")
        ax.set_ylabel("Velocità di crociera [kt]")
        forma = MODEL_FORM_PRESETS[nome]
        ax.set_title(nome if forma.is_baseline else f"{nome}: {forma.label}",
                     fontsize=10)

    presenti = sorted({int(v) for m in mappe.values() for v in np.unique(m)})
    handles = [Patch(color=colori[k + 1],
                     label=etichette[k] if k >= 0 else "nessuno fattibile")
               for k in presenti]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 4), fontsize=9)
    fig.suptitle("Sistema più efficiente con i parametri al valore nominale "
                 "(mappa deterministica)", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


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
    fig, axes = plt.subplots(1, len(FORMS),
                             figsize=(5.2 * len(FORMS), 4.4), squeeze=False)
    for ax, nome in zip(axes[0], FORMS):
        pm = aggregate_to_technologies(pmaps[nome]) if MAPPE_AGGREGATE else pmaps[nome]
        plot_probability_map(pm, ax=ax)
        forma = MODEL_FORM_PRESETS[nome]
        ax.set_title(nome if forma.is_baseline else f"{nome}: {forma.label}",
                     fontsize=10)
    fig.suptitle("Mappa probabilistica sotto le diverse forme del modello",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    maps_path = OUT_DIR / "model_form_maps.png"
    fig.savefig(maps_path, dpi=140)

    # dove le mappe non sono d'accordo: una cella accesa = lì la
    # variante cambia la tecnologia consigliata
    varianti = [nm for nm in FORMS if nm != "base"]
    fig, axes = plt.subplots(1, len(varianti),
                             figsize=(4.6 * len(varianti), 4.0), squeeze=False)
    vincitore_base = np.argmax(pmaps["base"].probabilities, axis=0)
    for ax, nome in zip(axes[0], varianti):
        diverso = (np.argmax(pmaps[nome].probabilities, axis=0) != vincitore_base)
        ax.pcolormesh(pmaps["base"].ranges_nmi, pmaps["base"].speeds_kt,
                      diverso.astype(float), cmap="Reds", vmin=0, vmax=1,
                      shading="auto")
        ax.set_xscale("log")
        ax.set_xlabel("range [nmi]")
        ax.set_ylabel("velocità [kt]")
        ax.set_title(f"{nome}: {diverso.mean():.1%} delle celle", fontsize=10)
    fig.suptitle("Dove la variante cambia la tecnologia consigliata "
                 "(rosso = decisione diversa dal modello base)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    dis_path = OUT_DIR / "model_form_disagreement.png"
    fig.savefig(dis_path, dpi=140)

    print(f"\nFigure salvate in {OUT_DIR}:")
    for p in (det_path, maps_path, dis_path):
        print(f"  {p.name}")

    plt.close("all")


if __name__ == "__main__":
    main()
