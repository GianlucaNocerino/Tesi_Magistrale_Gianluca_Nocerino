"""
Filiere reali di produzione: mappe del miglior sistema propulsivo,
rispetto all'energia primaria alla fonte e alla CO2 emessa.

Il modello deterministico assume che tutto passi dall'elettricità di
rete (rinnovabile) e quindi a monte non ci sono emissioni. Oggi
non è così: l'idrogeno si ottiene dal gas naturale, i SAF dalle
biomasse via HEFA, e solo la batteria continua davvero a dipendere dal
mix elettrico. Questo script sostituisce l'ipotesi "tutto elettrico"
con tre filiere reali, ciascuna riassunta da due parametri incerti, e
rifa' le mappe probabilistiche del sistema migliore usando due metriche
nuove al posto dell'electricity intensity:

  1. energia primaria richiesta alla fonte   [MJ/(pax*nmi)]
  2. anidride carbonica emessa               [g/(pax*nmi)]

NON SERVE RILANCIARE LA PROPAGAZIONE
--------------------------------------------
Nel modello l'efficienza well-to-tank del vettore energetico entra
soltanto nell'ultimo passaggio di compute_intensity: il dimensionamento
non la vede mai. L'energia effettivamente imbarcata per
passeggero-miglio quindi non dipende da nessun parametro di filiera, e
si ricostruisce esattamente dal cubo della propagazione già fatta prima:

    e_tank[k,j,i,r] = intensities[k,j,i,r] * eta_wtt_del_vettore[k]

dove eta_wtt è già una colonna di theta. Moltiplicando campione per
campione, l'incertezza della precedente efficienza di catena si cancella 
esattamente (non è un'approssimazione: è la stessa quantità che era stata 
divisa), e resta l'energia al serbatoio, che porta solo l'incertezza 
del velivolo.
Sopra ci si applicano le PDF delle filiere nuove.

Per la batteria il discorso è diverso: wtt.electricity è trasmissione e 
carica, quindi resta pertinente anche nello scenario reale. 
Per quel vettore l'intensity del cubo è già l'elettricità richiesta alla rete, 
e si usa direttamente.

Uso:
    python execution/11_aggiunte_well_to_tank/supply_chain_maps.py

Presuppone execution/07_propagazione/propagation_result.npz.

Output (nella cartella di questo script):
    supply_chain_specs.csv        la tabella delle PDF dei nuovi parametri
    map_co2.png                   mappa del migliore per CO2
    map_primary.png               mappa del migliore per energia primaria
    map_co2.csv, map_primary.csv  le probabilita' punto per punto

"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav.uncertainty import (CorrelationModel, ParameterSpec, PropagationResult,
                              classify_regions, plot_probability_map,
                              probability_map, robust_area_fraction,
                              sample_literature_parameters, triangular)

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR.parent / "07_propagazione" / "propagation_result.npz"

SEED = 12                 # seed del campionamento delle filiere: diverso da
                          # quello della Fase 7, i due campioni sono indipendenti
THRESHOLD = 0.90          # soglia di robustezza usata nelle mappe
CHECK_REGRESSIONE = True  # test di non regressione, vedi in fondo


# =====================================================================
# 1. INPUT - le tre filiere
# =====================================================================

# ---------------------------------------------------------------------
# Ogni filiera è riassunta da due soli parametri, che coprono l'intero
# well-to-tank, dalla fonte primaria fino al serbatoio dell'aeromobile:
#
#   eta   rendimento di catena: MJ di vettore energetico disponibili a
#         bordo per ogni MJ di energia primaria impiegata. L'energia
#         primaria richiesta è l'energia a bordo divisa per questo.
#
#   g     indice di emissione: grammi di CO2 per ogni MJ di vettore
#         energetico prodotto, cioè riferito all'energia che arriva a
#         bordo. Comprende tutto quello che si
#         decide di attribuire alla filiera (processo, trasporto,
#         liquefazione, combustione a bordo dove la si vuole contare).
#
# Per la filiera "grid" i due parametri hanno lo stesso significato ma
# riferiti al MJ elettrico prelevato dalla rete, perchè per la batteria
# il vettore energetico è l'elettricità.
#
# Entrambi sono parametri incerti: qui si dichiara uniforme
# (minimo, massimo), con minimo/massimo = moda +/- 5% moda
# ---------------------------------------------------------------------
FILIERE = {
    "grid": {
        "descrizione": "Elettricità di rete, mix europeo (batteria)",
        "eta": (0.487-0.05*0.487, 0.487, 0.487+0.05*0.487),
        "g":   (92.4-0.05*92.4, 92.4, 92.4+0.05*92.4),      # g/MJ elettrico
        "fonte_eta": "GREET",
        "fonte_g": "GREET",
    },
    "lh2": {
        "descrizione": "Idrogeno liquido da gas naturale (SMR + liquefazione)",
        "eta": (0.48-0.05*0.48, 0.48, 0.48+0.05*0.48),
        "g":   (112.55-0.05*112.55, 112.55, 112.55+0.05*112.55),     # g/MJ di LH2 al serbatoio
        "fonte_eta": "GREET",
        "fonte_g": "GREET",
    },
    "saf": {
        "descrizione": "SAF da biomasse via HEFA",
        "eta": (0.45712008-0.05*0.45712008, 0.45712008, 0.45712008+0.05*0.45712008),
        "g":   (31.832-0.05*31.832, 31.832, 31.832+0.05*31.832),      # g/MJ di SAF al serbatoio
        "fonte_eta": "GREET",
        "fonte_g": "GREET",
    },
}

# ---------------------------------------------------------------------
# Quale filiera alimenta quale tecnologia. La chiave e' il nome della
# tecnologia come compare nelle etichette del cubo, senza il propulsore.
#
# Nota: con la filiera HEFA il combustibile NON e' piu' un electrofuel.
# La tecnologia resta identica nel modello (stesso drop-in, stesso LHV) e
# l'etichetta "e-SAF combustion" viene mantenuta, perche' e' la chiave con
# cui technology_map assegna i colori: cambiarla qui romperebbe le figure
# e renderebbe le mappe non sovrapponibili a quelle della Fase 8. In
# questa fase quella voce va letta come "SAF da HEFA", e i titoli delle
# figure lo dicono esplicitamente.
# ---------------------------------------------------------------------
TECNOLOGIA_FILIERA = {
    "Battery-electric":    "grid",
    "Hydrogen fuel cell":  "lh2",
    "Hydrogen combustion": "lh2",
    "e-SAF combustion":    "saf",
}

# Quale colonna di theta contiene l'efficienza well-to-tank da rimuovere
# per tornare all'energia al serbatoio. Per la batteria e' None: il
# valore del cubo e' gia' l'elettricita' prelevata dalla rete e resta
# valido anche nello scenario reale.
TECNOLOGIA_WTT = {
    "Battery-electric":    None,
    "Hydrogen fuel cell":  "liquid_hydrogen",
    "Hydrogen combustion": "liquid_hydrogen",
    "e-SAF combustion":    "e_saf",
}

NOTA_FIGURE = "e-SAF combustion = SAF da HEFA in questa fase"


# =====================================================================
# 2. LE PDF DEI NUOVI PARAMETRI
# =====================================================================

def build_supply_chain_specs() -> dict:
    """Le PDF dei parametri di filiera, nello stesso formato della Fase 5.

    Sono deliberatamente TENUTE FUORI da build_default_specs(): aggiungerle
    lì cambierebbe la dimensione di theta e il disegno LHS, e i risultati
    delle Fasi 7, 9 e 10 non sarebbero piu' riproducibili.
    """
    specs = {}
    for nome, dati in FILIERE.items():
        a, m, b = dati["eta"]
        specs[f"eta_{nome}"] = ParameterSpec(
            name=f"eta_{nome}",
            dist=stats.uniform(loc=a, scale=b-a),
            nominal=m,
            pdf_label=f"Uniforme({a}, {b})",
            source=dati["fonte_eta"],
            rationale=f"Rendimento di catena well-to-tank: {dati['descrizione']}",
            mode=None,
        )
        a, m, b = dati["g"]
        specs[f"gco2_{nome}"] = ParameterSpec(
            name=f"gco2_{nome}",
            dist=stats.uniform(loc=a, scale=b-a),
            nominal=m,
            pdf_label=f"Uniforme({a}, {b})",
            source=dati["fonte_g"],
            rationale=f"Indice di emissione per MJ di vettore prodotto: {dati['descrizione']}",
            mode=None,
        )
    return specs


def sample_supply_chain(n_samples: int, specs: dict, seed: int = SEED) -> pd.DataFrame:
    """Campiona i parametri di filiera con lo stesso motore LHS della Fase 5.

    Nessuna correlazione imposta: le tre filiere sono impianti a terra
    distinti, e i loro rendimenti non condividono nessun fattore comune
    come invece facevano elettrolisi e liquefazione. Si tratta di un'assunzione, 
    e come tale va dichiarata.
    """
    return sample_literature_parameters(n_samples, specs=specs,
                                        correlations=CorrelationModel(),
                                        seed=seed)


# =====================================================================
# 3. DAL CUBO DELLA FASE 7 AI DUE CUBI NUOVI
# =====================================================================

def _colonna(theta: pd.DataFrame, nome: str) -> np.ndarray:
    if nome not in theta.columns:
        raise KeyError(
            f"theta non contiene la colonna '{nome}': il cubo caricato non è "
            f"stato prodotto con build_default_specs() (colonne presenti: "
            f"{list(theta.columns)})")
    return theta[nome].to_numpy(dtype=float)


def energia_al_serbatoio(result: PropagationResult) -> np.ndarray:
    """Ricostruisce l'energia imbarcata per pax*nmi, (N, 8, n_speeds, n_ranges).

    Per la batteria restituisce invece l'elettricità prelevata dalla
    rete, che è già il valore del cubo: è il punto di partenza giusto
    per quella filiera.
    """
    cubo = np.array(result.intensities, dtype=float)
    for j, label in enumerate(result.labels):
        tecnologia = label.rsplit(" (", 1)[0]
        colonna = TECNOLOGIA_WTT[tecnologia]
        if colonna is None:
            continue
        eta_wtt = _colonna(result.theta, colonna)
        cubo[:, j, :, :] *= eta_wtt[:, None, None]
    return cubo


def cubi_primaria_e_co2(result: PropagationResult, theta_sc: pd.DataFrame) -> tuple:
    """Applica le filiere e restituisce (cubo_primaria, cubo_co2)"""
    base = energia_al_serbatoio(result)
    primaria = np.empty_like(base)
    co2 = np.empty_like(base)

    for j, label in enumerate(result.labels):
        tecnologia = label.rsplit(" (", 1)[0]
        filiera = TECNOLOGIA_FILIERA[tecnologia]
        eta = theta_sc[f"eta_{filiera}"].to_numpy(dtype=float)[:, None, None]
        g = theta_sc[f"gco2_{filiera}"].to_numpy(dtype=float)[:, None, None]
        primaria[:, j, :, :] = base[:, j, :, :] / eta
        co2[:, j, :, :] = base[:, j, :, :] * g

    return primaria, co2


def _best_index(cubo: np.ndarray) -> np.ndarray:
    """L'etichetta che minimizza la metrica, -1 dove nessuna è fattibile.

    Stessa convenzione di run_propagation: il NaN significa "non
    fattibile" e non deve mai vincere.
    """
    finito = np.isfinite(cubo)
    qualcuno = finito.any(axis=1)
    sicuro = np.where(finito, cubo, np.inf)
    idx = np.argmin(sicuro, axis=1)
    return np.where(qualcuno, idx, -1).astype(np.int8)


def risultato_con_metrica(result: PropagationResult, cubo: np.ndarray) -> PropagationResult:
    """Impacchetta un cubo di metrica in un PropagationResult.

    Il passaggio che rende gratuito tutto il resto: probability_map,
    classify_regions, robust_area_fraction e i plot leggono solo
    best_index, labels e grid, quindi funzionano su questa metrica
    esattamente come sull'electricity intensity, senza modifiche.
    """
    return PropagationResult(
        intensities=cubo,
        best_index=_best_index(cubo),
        theta=result.theta,
        grid=result.grid,
        labels=list(result.labels),
    )


# =====================================================================
# 4. MAPPE E STAMPA
# =====================================================================

def disegna_mappe(res: PropagationResult, titolo: str, slug: str) -> None:
    pmap = probability_map(res)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_probability_map(pmap, threshold=THRESHOLD, ax=ax,
                         title=f"{titolo} (N = {pmap.n_samples})\n{NOTA_FIGURE}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"map_{slug}.png", dpi=150)
    plt.close(fig)

    pmap.to_dataframe().to_csv(OUT_DIR / f"map_{slug}.csv", index=False)

    # quanto è netta la mappa: frazione di piano in cui una tecnologia
    # vince con probabilità sopra soglia. classify_regions restituisce
    # l'indice della tecnologia robusta, -1 dove nessuna lo è
    regioni = classify_regions(pmap, threshold=THRESHOLD)
    print(f"\n{titolo}")
    print(f"  area robusta (P >= {THRESHOLD:.2f}): "
          f"{100 * float(np.mean(regioni >= 0)):.1f}% del piano")
    for j, label in enumerate(pmap.labels):
        frazione = float(np.mean(regioni == j))
        if frazione > 0:
            print(f"    {label:<28}{100 * frazione:>6.1f}%")
    print("  ripartizione al variare della soglia:")
    print(robust_area_fraction(pmap).round(3).to_string(index=False))


def check_regressione(result: PropagationResult) -> None:
    """Test di non regressione.

    Se si usa una sola filiera per tutti i vettori partendo
    dall'electricity intensity, i due nuovi output sono l'intensity
    moltiplicata per due costanti uguali per tutte le etichette: il
    vincitore non può cambiare, e le mappe devono coincidere con quelle
    della Fase 8. Se questo controllo fallisce, l'errore è nell'algebra
    della ricostruzione, non nei dati di filiera
    """
    cubo = np.array(result.intensities, dtype=float)
    uguale_co2 = np.array_equal(_best_index(cubo * 46.5), result.best_index)
    uguale_pri = np.array_equal(_best_index(cubo / 0.55), result.best_index)
    esito = "OK" if (uguale_co2 and uguale_pri) else "FALLITO"
    print(f"\nTest di non regressione (scenario tutto-elettrico): {esito}")


# =====================================================================
# 5. MAIN
# =====================================================================

def main() -> None:
    result = PropagationResult.load(RESULT_PATH)
    print(f"Caricati {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")

    specs = build_supply_chain_specs()
    theta_sc = sample_supply_chain(result.n_samples, specs)

    tabella = pd.DataFrame([
        {"parametro": s.name, "nominale": s.nominal, "PDF": s.pdf_label,
         "fonte": s.source, "motivazione": s.rationale}
        for s in specs.values()
    ])
    print("\n" + "=" * 78)
    print("PARAMETRI DI FILIERA")
    print("=" * 78)
    print(tabella[["parametro", "nominale", "PDF"]].to_string(index=False))
    tabella.to_csv(OUT_DIR / "supply_chain_specs.csv", index=False)

    primaria, co2 = cubi_primaria_e_co2(result, theta_sc)

    res_co2 = risultato_con_metrica(result, co2)
    res_pri = risultato_con_metrica(result, primaria)

    disegna_mappe(res_co2, "Sistema con la minima CO2", "co2")
    disegna_mappe(res_pri, "Sistema con la minima energia 'alle fonti'", "primary")

    # quanto le due metriche sono d'accordo fra loro: dove non lo sono,
    # c'e' un compromesso fra emissioni e domanda di energia primaria
    valide = (res_co2.best_index >= 0) & (res_pri.best_index >= 0)
    accordo = float(np.mean(res_co2.best_index[valide] == res_pri.best_index[valide]))
    print(f"\nLe due metriche indicano lo stesso vincitore nel {100 * accordo:.1f}% "
          f"delle celle (campione, punto di griglia)")

    if CHECK_REGRESSIONE:
        check_regressione(result)

    print(f"\nFigure e tabelle salvate in: {OUT_DIR}")


if __name__ == "__main__":
    main()
