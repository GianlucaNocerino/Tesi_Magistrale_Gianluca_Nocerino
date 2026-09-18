"""
Filiere reali di produzione: mappe del miglior sistema propulsivo,
rispetto all'energia primaria alla fonte e alla CO2 emessa, per lo
scenario attuale e per gli scenari 2035 e 2050.

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

La stessa analisi si ripete per più scenari temporali (attuale, 2035,
2050): cambiano solo le PDF dei parametri di filiera, e se necessario
la descrizione della filiera stessa (es. idrogeno da elettrolisi al
posto dello SMR).

I SEI PARAMETRI NON SONO INDIPENDENTI
--------------------------------------------
Nel 2035 e nel 2050 i tre punti di ogni terna differiscono solo per il
mix elettrico: sono tre scenari IEA 2025, e per ciascuno rendimenti ed
emissioni di ogni processo escono da GREET alimentato con quel mix. I
sei parametri sono quindi sei immagini della stessa variabile, e
campionarli come indipendenti produrrebbe combinazioni che non esistono, 
es: rete decarbonizzata con e-SAF sporchissimo. La dipendenza è imposta con
copule gaussiane sui quantili, quindi senza toccare le marginali: vedi
ACCOPPIAMENTO, più sotto, che spiega anche perchè nello scenario attuale
il legame è di natura diversa.

ATTENZIONE: il cubo della Fase 7 descrive i velivoli con le tecnologie
di bordo campionate in quella fase. Negli scenari 2035 e 2050 cambia
solo la filiera a terra: se le PDF della Fase 5 non rappresentano già
le tecnologie di quegli anni, le mappe future combinano filiere future
con velivoli di oggi, e vanno lette come tali.

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
        -> tutti gli scenari gia' compilati (quelli incompleti vengono saltati)
    python execution/11_aggiunte_well_to_tank/supply_chain_maps.py 2035 2050
        -> solo gli scenari indicati

Presuppone execution/07_propagazione/propagation_result.npz.

Output (nella cartella di questo script), con <s> = attuale, 2035, 2050:
    supply_chain_specs_<s>.csv        la tabella delle PDF dei nuovi parametri
    map_co2_<s>.png                   mappa del migliore per CO2
    map_primary_<s>.png               mappa del migliore per energia primaria
    map_co2_<s>.csv, map_primary_<s>.csv   le probabilità punto per punto
    riepilogo_scenari.csv             aree robuste e accordo fra metriche, per scenario
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav.uncertainty import (CopulaPair, CorrelationModel, ParameterSpec,
                              PropagationResult, TriangularWithFloor,
                              classify_regions, plot_probability_map,
                              probability_map, robust_area_fraction,
                              sample_literature_parameters)

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR.parent / "07_propagazione" / "propagation_result.npz"

SEED = 12                 # seed del campionamento delle filiere: diverso da
                          # quello della Fase 7, i due campioni sono indipendenti.
                          # E' lo STESSO per tutti gli scenari (common random
                          # numbers): con gli stessi parametri nello stesso
                          # ordine, l'LHS estrae gli stessi quantili, e le
                          # differenze fra scenari dipendono solo dalle PDF,
                          # non dal rumore di campionamento
THRESHOLD = 0.90          # soglia di robustezza usata nelle mappe
CHECK_REGRESSIONE = True  # test di non regressione, vedi in fondo
FONDO_PDF = 0.18          # peso dell'uniforme nelle PDF dei parametri di
                          # filiera: è quanto le distribuzioni vengono
                          # sollevate da zero agli estremi. Vedi
                          # TriangularWithFloor in distributions.py


# =====================================================================
# 1. INPUT - le tre filiere, per scenario
# =====================================================================

# ---------------------------------------------------------------------
# Ogni filiera è riassunta da due soli parametri, che coprono l'intero
# well-to-tank, dalla fonte primaria fino al serbatoio dell'aeromobile:
#
#   eta   rendimento di catena: MJ di vettore energetico disponibili a
#         bordo per ogni MJ di energia primaria impiegata. L'energia
#         primaria richiesta è l'energia a bordo divisa per questo.
#         Può superare l'unità: con la convenzione usata qui per
#         l'energia primaria delle rinnovabili, una rete molto
#         decarbonizzata ha un rendimento di catena maggiore di 1, e la
#         validazione non impone quindi nessun limite superiore.
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
# Entrambi sono parametri incerti con PDF triangolare sollevata agli
# estremi, dichiarata come terna (minimo, moda, massimo). La terna può
# essere asimmetrica. Per lo scenario attuale si mantiene l'intervallo
# moda +/- 5% moda; per 2035 e 2050 i tre valori vengono ciascuno da
# uno dei tre scenari IEA 2025 (vedi ACCOPPIAMENTO, che è la
# conseguenza di questo fatto sulla struttura di dipendenza).
#
# Se anche una sola terna vale None, lo scenario viene saltato con un
# avviso.
# ---------------------------------------------------------------------

ORDINE_FILIERE = ("grid", "lh2", "saf")   # ordine fisso: garantisce che
                                          # theta_sc abbia le stesse colonne,
                                          # nello stesso ordine, in ogni scenario


def intorno(moda: float, frazione: float = 0.05) -> tuple:
    """Terna (minimo, moda, massimo) simmetrica: moda +/- frazione*moda."""
    return (moda * (1 - frazione), moda, moda * (1 + frazione))


SCENARI = {
    "attuale": {
        "titolo": "scenario attuale",
        "filiere": {
            "grid": {
                "descrizione": "Elettricità di rete, mix globale (batteria)",
                "eta": intorno(0.457875458),
                "g":   intorno(140.26),          # g/MJ elettrico
                "fonte_eta": "GREET",
                "fonte_g": "GREET",
            },
            "lh2": {
                "descrizione": "Idrogeno liquido da gas naturale (SMR + liquefazione)",
                "eta": intorno(0.463073),
                "g":   intorno(126.7482),        # g/MJ di LH2 al serbatoio
                "fonte_eta": "GREET",
                "fonte_g": "GREET",
            },
            "saf": {
                "descrizione": "SAF da biomasse via HEFA",
                "eta": intorno(0.455974),
                "g":   intorno(32.0),        # g/MJ di SAF al serbatoio
                "fonte_eta": "GREET",
                "fonte_g": "GREET",
            },
        },
    },

    "2035": {
        "titolo": "scenario 2035",
        "filiere": {
            "grid": {
                "descrizione": "Elettricità di rete, mix 2035 (batteria)",
                "eta": (0.5773672055, 0.6199628022, 0.8865248227),
                "g":   (29.1, 85.1, 97.8),                  # g/MJ elettrico
                "fonte_eta": "",
                "fonte_g": "",
            },
            "lh2": {
                "descrizione": "Idrogeno liquido, filiera 2035",
                "eta": (0.40799025, 0.4378, 0.538794),
                "g":   (67.20345, 129.02795, 145.5304),     # g/MJ di LH2 al serbatoio
                "fonte_eta": "",
                "fonte_g": "",
            },
            "saf": {
                "descrizione": "SAF, filiera 2035",
                "eta": (0.367158, 0.3690775, 0.388303625),
                "g":   (66.294175, 147.291225, 165.616075),  # g/MJ di SAF al serbatoio
                "fonte_eta": "",
                "fonte_g": "",
            },
        },
    },

    "2050": {
        "titolo": "scenario 2050",
        "filiere": {
            "grid": {
                "descrizione": "Elettricità di rete, mix 2050 (batteria)",
                "eta": (0.69881202, 0.8417508418, 1.2121212),
                "g":   (5.9, 45.4, 68.9),
                "fonte_eta": "",
                "fonte_g": "",
            },
            "lh2": {
                "descrizione": "Idrogeno liquido, filiera 2050",
                "eta": (0.379472, 0.45692, 0.6565),
                "g":   (11.5094, 83.8563, 127.0911),
                "fonte_eta": "",
                "fonte_g": "",
            },
            "saf": {
                "descrizione": "SAF, filiera 2050",
                "eta": (0.2044695, 0.222913351, 0.27032025),
                "g":   (35.3564, 206.0585, 308.071),
                "fonte_eta": "",
                "fonte_g": "",
            },
        },
    },
}

# ---------------------------------------------------------------------
# LA DIPENDENZA FRA I SEI PARAMETRI
#
# Due legami, entrambi imposti con copule gaussiane sui quantili, quindi
# senza toccare le marginali dichiarate sopra:
#
#   dentro la filiera   rendimento e indice di emissione si muovono in
#                       verso opposto: la catena che rende di più emette
#                       di meno per MJ consegnato a bordo. RHO_ETA_G.
#
#   verso la rete       idrogeno ed e-fuel elettrolitici ereditano il
#                       mix elettrico che li alimenta, quindi le loro
#                       emissioni seguono quelle della rete.
#
# ACCOPPIAMENTO dice quanto forte è il secondo legame, e i due valori
# rispondono a due situazioni completamente diverse:
#
#   "scenario"  (2035, 2050) dentro l'anno il processo è fissato, e fra i
#               tre punti della terna cambia solo il mix elettrico: i tre
#               scenari IEA 2025 danno tre mix, e ogni valore di
#               rendimento ed emissione esce da GREET alimentato con
#               quel mix. I sei parametri sono quindi sei immagini della
#               stessa variabile, e la dipendenza di rango perfetto
#               (rho = 1) non è un'ipotesi ma la descrizione di come i
#               numeri sono stati costruiti.
#
#               Importante: la quota elettrica della filiera non entra
#               qui come peso. La sensibilità al mix è già dentro
#               l'ampiezza della terna, una filiera poco elettrificata ha
#               tre letture GREET vicine fra loro. Pesare anche il rho
#               conterebbe la stessa cosa due volte.
#
#   "quota"     (attuale) non ci sono scenari, c'è un dato, e la terna è
#               una banda di incertezza attorno a quel dato. Quelle
#               fluttuazioni sono in buona parte proprie di ciascuna
#               filiera; la parte condivisa è solo quella che passa per
#               l'elettricità di rete, e vale quanto QUOTA_ELETTRICA.
#
# QUOTA_ELETTRICA serve quindi al solo scenario attuale, per quantificare
# quanto l'elettrcità entra nei processi attuali
#
# VERSO_PULITO dichiara come gli scenari si dispongono sulla terna: +1 se
# il mix più pulito dà il rendimento più alto e l'emissione più bassa,
# cioè se le due terne della filiera sono ordinate in verso opposto, che
# è il caso dei numeri qui sopra. Sapendo quale scenario ha prodotto
# ciascun estremo, è una cosa da verificare e non da assumere.
# ---------------------------------------------------------------------
QUOTA_ELETTRICA = {          # solo per lo scenario attuale
    "lh2": 0.38,             # liquefazione dell'idrogeno
    "saf": 0.09,             # elettricità per processo HEFA
}

ACCOPPIAMENTO = {
    "attuale": "quota",
    "2035": "scenario",
    "2050": "scenario",
}

RHO_ETA_G = {                # dentro la filiera, per scenario
    "attuale": -0.7,         # banda generica: il legame c'è ma non è
                             # deterministico, parte dell'incertezza è
                             # rumore proprio di ciascun parametro
    "2035": -1.0,            # stessa variabile latente (il mix): rango
    "2050": -1.0,            # perfetto e opposto
}

VERSO_PULITO = {"grid": +1, "lh2": +1, "saf": +1}

# Solo documentazione: che processi ci sono dentro ogni filiera, anno per
# anno. Non entra nei calcoli, ma è l'informazione che spiega perchè le
# terne del 2050 sono così larghe e quelle del 2024 così strette.
PROCESSI = {
    "attuale": {"lh2": "100% SMR + liquefazione", "saf": "100% HEFA"},
    "2035": {"lh2": "50% SMR / 50% elettrolisi",
             "saf": "75% HEFA / 25% Fischer-Tropsch"},
    "2050": {"lh2": "100% elettrolisi",
             "saf": "25% HEFA / 75% Fischer-Tropsch"},
}

# ---------------------------------------------------------------------
# Quale filiera alimenta quale tecnologia. La chiave è il nome della
# tecnologia come compare nelle etichette del cubo, senza il propulsore.
#
# Nota: il combustibile "saf" può non essere un electrofuel (oggi è
# HEFA). La tecnologia resta identica nel modello (stesso drop-in, stesso
# LHV) e l'etichetta "e-SAF combustion" viene mantenuta, perchè è la
# chiave con cui technology_map assegna i colori: cambiarla qui romperebbe
# le figure e renderebbe le mappe non sovrapponibili a quelle della
# Fase 8. In ogni scenario quella voce va letta secondo la descrizione
# della filiera "saf", che i titoli delle figure riportano esplicitamente.
# ---------------------------------------------------------------------
TECNOLOGIA_FILIERA = {
    "Battery-electric":    "grid",
    "Hydrogen fuel cell":  "lh2",
    "Hydrogen combustion": "lh2",
    "e-SAF combustion":    "saf",
}

# Quale colonna di theta contiene l'efficienza well-to-tank da rimuovere
# per tornare all'energia al serbatoio. Per la batteria è None: il
# valore del cubo è già l'elettricita' prelevata dalla rete e resta
# valido anche nello scenario reale.
TECNOLOGIA_WTT = {
    "Battery-electric":    None,
    "Hydrogen fuel cell":  "liquid_hydrogen",
    "Hydrogen combustion": "liquid_hydrogen",
    "e-SAF combustion":    "e_saf",
}


def nota_figure(scenario: dict) -> str:
    return f"e-SAF combustion = {scenario['filiere']['saf']['descrizione']}"


# =====================================================================
# 2. LE PDF DEI NUOVI PARAMETRI E LA LORO DIPENDENZA
# =====================================================================

def scenario_compilato(scenario: dict) -> bool:
    """True se tutte le terne dello scenario sono state inserite"""
    filiere = scenario["filiere"]
    return all(
        nome in filiere
        and filiere[nome].get("eta") is not None
        and filiere[nome].get("g") is not None
        for nome in ORDINE_FILIERE
    )


def _valida_terna(terna, dove: str, limiti: tuple = None) -> tuple:
    """Controlla una terna (minimo, moda, massimo) e la restituisce in float.

    I controlli su ordine e degenerazione ripetono quelli di triangular(),
    ma qui il messaggio indica scenario, filiera e parametro.
    limiti = (lo, hi) impone lo < minimo e massimo <= hi; hi = None
    significa nessun limite superiore, che è il caso del rendimento di
    catena (vedi il commento sulla convenzione dell'energia primaria).
    """
    try:
        a, m, b = (float(x) for x in terna)
    except (TypeError, ValueError):
        raise ValueError(f"{dove}: serve una terna (minimo, moda, massimo), "
                         f"trovato {terna!r}") from None
    if not a <= m <= b:
        raise ValueError(f"{dove}: deve valere minimo <= moda <= massimo, "
                         f"trovato {terna}")
    if a == b:
        raise ValueError(f"{dove}: minimo e massimo coincidono, "
                         f"la triangolare degenera")
    if limiti is not None:
        lo, hi = limiti
        if a <= lo:
            raise ValueError(f"{dove}: il minimo deve essere maggiore di {lo}, "
                             f"trovato {terna}")
        if hi is not None and b > hi:
            raise ValueError(f"{dove}: il massimo non può superare {hi}, "
                             f"trovato {terna}")
    return a, m, b


def build_supply_chain_specs(filiere: dict, scenario: str) -> dict:
    """Le PDF dei parametri di filiera, nello stesso formato della Fase 5.

    Sono deliberatamente tenute fuori da build_default_specs(): aggiungerle
    lì cambierebbe la dimensione di theta e il disegno LHS, e i risultati
    delle Fasi 7, 9 e 10 non sarebbero piu' riproducibili.

    I nomi dei parametri non contengono lo scenario: sono gli stessi in
    ogni scenario, cosi' il campionamento resta allineato (vedi SEED).
    """
    parametri = (
        # chiave, prefisso, campo fonte, descrizione, limiti ammessi
        ("eta", "eta",  "fonte_eta", "Rendimento di catena well-to-tank",              (0.0, None)),
        ("g",   "gco2", "fonte_g",   "Indice di emissione per MJ di vettore prodotto", None),
    )
    specs = {}
    for nome in ORDINE_FILIERE:
        dati = filiere[nome]
        for chiave, prefisso, campo_fonte, cosa, limiti in parametri:
            a, m, b = _valida_terna(dati[chiave], f"[{scenario}] {nome}.{chiave}", limiti)
            specs[f"{prefisso}_{nome}"] = ParameterSpec(
                name=f"{prefisso}_{nome}",
                dist=TriangularWithFloor(a, m, b, FONDO_PDF),
                nominal=m,
                pdf_label=f"Triangolare({a:.6g}, {m:.6g}, {b:.6g}) + fondo {FONDO_PDF:.2f}",
                source=dati[campo_fonte],
                rationale=f"{cosa} [{scenario}]: {dati['descrizione']}",
                mode=m,
            )
    return specs


def build_supply_chain_correlations(chiave: str) -> CorrelationModel:
    """Le copule che legano i sei parametri, secondo QUOTA_ELETTRICA e
    ACCOPPIAMENTO.

    Il driver è gco2_grid, cioè "quanto è pulita la rete": da lì
    scendono, nell'ordine, l'emissione di ciascuna filiera e, dentro
    ogni filiera, il suo rendimento. L'ordine conta, perchè le copule
    sono applicate in sequenza sui quantili e ciascun parametro deve
    comparire una sola volta come secondo membro.

    La correlazione implicata fra due parametri non legati direttamente
    è il prodotto dei rho lungo il cammino: con rho = 1 fra rete e
    idrogeno e rho = -1 dentro la filiera, il rendimento dell'idrogeno
    risulta anticorrelato con l'emissione della rete, che è quello che
    si vuole.
    """
    modo = ACCOPPIAMENTO[chiave]
    if modo not in ("quota", "scenario"):
        raise ValueError(f"ACCOPPIAMENTO['{chiave}'] = {modo!r}: "
                         f"ammessi solo 'quota' e 'scenario'")
    rho_interno = float(RHO_ETA_G[chiave])
    coppie = []

    # dentro la rete: rendimento contro emissione
    coppie.append(CopulaPair(
        a="gco2_grid", b="eta_grid",
        rho=VERSO_PULITO["grid"] * rho_interno,
        rationale="Rete: più il mix è pulito, più alto è il rendimento di catena"))

    for nome in ORDINE_FILIERE:
        if nome == "grid":
            continue
        peso = 1.0 if modo == "scenario" else float(QUOTA_ELETTRICA[nome])
        peso = float(np.clip(peso, 0.0, 1.0))
        if peso > 0.0:
            motivo = ("stesso mix elettrico dietro i tre punti della terna"
                      if modo == "scenario"
                      else f"quota elettrica della filiera = {peso:.2f}")
            coppie.append(CopulaPair(
                a="gco2_grid", b=f"gco2_{nome}", rho=peso,
                rationale=f"{nome}: {motivo}"))
        coppie.append(CopulaPair(
            a=f"gco2_{nome}", b=f"eta_{nome}",
            rho=VERSO_PULITO[nome] * rho_interno,
            rationale=f"{nome}: rendimento ed emissione si muovono in verso opposto"))

    return CorrelationModel(copulas=tuple(coppie))


def sample_supply_chain(n_samples: int, specs: dict,
                        correlations: CorrelationModel,
                        seed: int = SEED) -> pd.DataFrame:
    """Campiona i parametri di filiera con lo stesso motore LHS della Fase 5.

    Le copule agiscono sui quantili prima della trasformata inversa,
    quindi le marginali restano esattamente quelle dichiarate in
    SCENARI: cambia solo come i sei parametri si muovono insieme.
    """
    return sample_literature_parameters(n_samples, specs=specs,
                                        correlations=correlations,
                                        seed=seed)


def stampa_correlazioni(theta_sc: pd.DataFrame,
                        correlations: CorrelationModel) -> None:
    """Correlazioni realizzate nel campione contro quelle volute.

    Stesso controllo della Fase 7: quello che si voleva imporre va
    verificato, non dato per fatto. I valori sono di Pearson sui valori
    fisici, mentre la copula impone la dipendenza sui ranghi: con
    marginali molto asimmetriche uno scarto di qualche centesimo da
    |rho| = 1 è normale e non è un errore.
    """
    print("  correlazioni realizzate (target fra parentesi):")
    for pair in correlations.copulas:
        r = theta_sc[pair.a].corr(theta_sc[pair.b])
        print(f"    {pair.a:<12} / {pair.b:<12} r = {r:+.3f}  ({pair.rho:+.2f})")


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


def cubi_primaria_e_co2(base: np.ndarray, labels: list, theta_sc: pd.DataFrame) -> tuple:
    """Applica le filiere all'energia al serbatoio e restituisce
    (cubo_primaria, cubo_co2).

    base e' l'output di energia_al_serbatoio: non dipende dallo scenario,
    quindi si calcola una volta sola e si riusa.
    """
    primaria = np.empty_like(base)
    co2 = np.empty_like(base)

    for j, label in enumerate(labels):
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

def disegna_mappe(res: PropagationResult, titolo: str, slug: str, nota: str) -> dict:
    """Salva mappa e CSV in OUT_DIR, stampa il riassunto e lo restituisce.

    slug contiene gia' lo scenario (es. "co2_2035"), cosi' i file dei
    diversi scenari convivono nella stessa cartella senza sovrascriversi.
    """
    pmap = probability_map(res)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_probability_map(pmap, threshold=THRESHOLD, ax=ax,
                         title=f"{titolo} (N = {pmap.n_samples})\n{nota}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"map_{slug}.png", dpi=150)
    plt.close(fig)

    pmap.to_dataframe().to_csv(OUT_DIR / f"map_{slug}.csv", index=False)

    # quanto è netta la mappa: frazione di piano in cui una tecnologia
    # vince con probabilità sopra soglia. classify_regions restituisce
    # l'indice della tecnologia robusta, -1 dove nessuna lo è
    regioni = classify_regions(pmap, threshold=THRESHOLD)
    area = 100 * float(np.mean(regioni >= 0))
    riepilogo = {"area_robusta_%": area}

    print(f"\n{titolo}")
    print(f"  area robusta (P >= {THRESHOLD:.2f}): {area:.1f}% del piano")
    for j, label in enumerate(pmap.labels):
        frazione = 100 * float(np.mean(regioni == j))
        riepilogo[f"{label} %"] = frazione
        if frazione > 0:
            print(f"    {label:<28}{frazione:>6.1f}%")
    print("  ripartizione al variare della soglia:")
    print(robust_area_fraction(pmap).round(3).to_string(index=False))

    return riepilogo


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
# 5. UNO SCENARIO
# =====================================================================

def esegui_scenario(chiave: str, scenario: dict, result: PropagationResult,
                    base: np.ndarray) -> list:
    """Campiona, costruisce i cubi e disegna le mappe di uno scenario.

    Restituisce le righe del riepilogo (una per metrica).
    """
    titolo = scenario["titolo"]
    nota = nota_figure(scenario)

    specs = build_supply_chain_specs(scenario["filiere"], chiave)
    correlazioni = build_supply_chain_correlations(chiave)
    theta_sc = sample_supply_chain(result.n_samples, specs, correlazioni)

    tabella = pd.DataFrame([
        {"scenario": chiave, "parametro": s.name, "nominale": s.nominal,
         "PDF": s.pdf_label, "fonte": s.source, "motivazione": s.rationale}
        for s in specs.values()
    ])
    print("\n" + "=" * 78)
    print(f"PARAMETRI DI FILIERA - {titolo.upper()}")
    print("=" * 78)
    print(tabella[["parametro", "nominale", "PDF"]].to_string(index=False))
    processi = ", ".join(f"{k}: {v}" for k, v in PROCESSI[chiave].items())
    print(f"\nFiliere: {processi}")
    print(f"Dipendenza: accoppiamento '{ACCOPPIAMENTO[chiave]}', "
          f"rho dentro la filiera {RHO_ETA_G[chiave]:+.2f}")
    stampa_correlazioni(theta_sc, correlazioni)
    tabella.to_csv(OUT_DIR / f"supply_chain_specs_{chiave}.csv", index=False)

    primaria, co2 = cubi_primaria_e_co2(base, result.labels, theta_sc)

    res_co2 = risultato_con_metrica(result, co2)
    res_pri = risultato_con_metrica(result, primaria)

    riep_co2 = disegna_mappe(res_co2, f"Sistema con la minima CO2 - {titolo}",
                             f"co2_{chiave}", nota)
    riep_pri = disegna_mappe(res_pri, f"Sistema con la minima energia 'alle fonti' - {titolo}",
                             f"primary_{chiave}", nota)

    # quanto le due metriche sono d'accordo fra loro: dove non lo sono,
    # c'e' un compromesso fra emissioni e domanda di energia primaria
    valide = (res_co2.best_index >= 0) & (res_pri.best_index >= 0)
    accordo = 100 * float(np.mean(res_co2.best_index[valide] == res_pri.best_index[valide]))
    print(f"\n[{chiave}] Le due metriche indicano lo stesso vincitore nel "
          f"{accordo:.1f}% delle celle (campione, punto di griglia)")

    comune = {"scenario": chiave, "accordo_metriche_%": accordo}
    return [
        {**comune, "metrica": "co2", **riep_co2},
        {**comune, "metrica": "primaria", **riep_pri},
    ]


# =====================================================================
# 6. MAIN
# =====================================================================

def main(argv: list = None) -> None:
    richiesti = list(argv) if argv else list(SCENARI)
    sconosciuti = [s for s in richiesti if s not in SCENARI]
    if sconosciuti:
        raise SystemExit(f"Scenari sconosciuti: {sconosciuti}. "
                         f"Disponibili: {list(SCENARI)}")

    result = PropagationResult.load(RESULT_PATH)
    print(f"Caricati {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")

    if CHECK_REGRESSIONE:
        check_regressione(result)

    # l'energia al serbatoio non dipende dalla filiera: una volta sola
    base = energia_al_serbatoio(result)

    righe = []
    for chiave in richiesti:
        scenario = SCENARI[chiave]
        if not scenario_compilato(scenario):
            print(f"\n[!] Scenario '{chiave}' saltato: ci sono terne "
                  f"(minimo, moda, massimo) ancora da compilare")
            continue
        righe += esegui_scenario(chiave, scenario, result, base)

    if righe:
        riepilogo = pd.DataFrame(righe).fillna(0.0)
        colonne = ["scenario", "metrica", "area_robusta_%", "accordo_metriche_%"]
        colonne += [c for c in riepilogo.columns if c not in colonne]
        riepilogo = riepilogo[colonne]
        riepilogo.to_csv(OUT_DIR / "riepilogo_scenari.csv", index=False)
        print("\n" + "=" * 78)
        print("RIEPILOGO SCENARI")
        print("=" * 78)
        print(riepilogo.round(1).to_string(index=False))
    else:
        print("\nNessuno scenario eseguito.")

    print(f"\nFigure e tabelle salvate in: {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])