"""
Effetti dell'energy mix: dall'elettricità alla fonte alle emissioni di
CO2 e all'energia primaria, con il cherosene convenzionale come termine
di paragone, sopra il campione della propagazione della Fase 7.

Il modello deterministico si ferma all'elettricità richiesta alla
fonte, assumendola implicitamente rinnovabile e quindi a emissioni
nulle. Questo script aggiunge lo strato che sta ancora più a monte:
dato un energy mix, riassunto da un rendimento di catena e da
un'intensità di carbonio, entrambi già mediati su tutte le fonti
(rinnovabili e non), calcola due nuovi output per passeggero-miglio
nautico:

  1. energia primaria richiesta          [MJ/(pax*nmi)]
  2. anidride carbonica emessa           [g/(pax*nmi)]

Rispetto alla versione precedente cambiano tre cose.

INCERTEZZA SUI PARAMETRI DI MIX
--------------------------------------------
Rendimento di catena e intensità di carbonio non sono più due numeri
ma due variabili aleatorie con PDF triangolare (minimo, moda, massimo),
campionate con lo stesso motore LHS delle Fasi 5 e 11. Anche la filiera
del cherosene porta la sua incertezza.

SI PARTE DAL CUBO DELLA FASE 7, NON DAL MODELLO NOMINALE
--------------------------------------------
I parametri di mix entrano soltanto nell'ultimo prodotto: il
dimensionamento non li vede mai. L'elettricità richiesta alla fonte per
pax*nmi e' quindi esattamente intensities[k,j,i,r] del cubo gia'
propagato, e i due nuovi output si ottengono campione per campione:

    primaria = intensities / eta_catena
    co2      = intensities * gco2_el

Così i risultati portano anche l'incertezza del velivolo, non solo
quella del mix, e non si ridimensiona nulla.

IL CHEROSENE ENTRA COME CONCORRENTE
--------------------------------------------
Il cherosene non è fra gli otto sistemi propagati nella Fase 7, e non
passa per l'elettricità: la sua catena parte dal greggio e arriva al
serbatoio, più la combustione a bordo. Va quindi dimensionato a parte,
sulla stessa griglia e sulle stesse righe di theta (la cellula è quella
dell'e-SAF, quindi condivide L/D, OEW e rendimenti: usare un campione
indipendente spezzerebbe la correlazione e allargherebbe gli intervalli
per finta). Costituisce l'unico calcolo pesante di questo script, e 
viene messo in cache su disco: si paga una volta sola.

Con il cherosene in gioco le mappe probabilistiche diventano
informative. Senza, sarebbero identiche a quella dell'electricity
intensity: eta_catena e gco2_el sono gli stessi per tutti i sistemi
sostenibili, quindi le due metriche sono trasformazioni monotone
dell'intensity con la stessa costante e argmin non cambia (questo lo
verifica check_regressione). Il cherosene invece non scala con quei
parametri, e sposta davvero la frontiera.

Uso:
    python execution/11_aggiunte_well_to_tank/energy_mix_effects.py

Presuppone execution/07_propagazione/propagation_result.npz.
La prima esecuzione propaga il cherosene: con 1500 campioni sulla
griglia 40x45 sono circa 45 minuti con N_WORKERS = 1, molto meno
aumentando i worker. Conviene provare prima con N_CAMPIONI piccolo.

Output (nella cartella di questo script):
    kerosene_cube.npz          cache della propagazione del cherosene
    energy_mix_specs.csv       le PDF dei parametri incerti
    map_co2.png                mappa del sistema con la minima CO2
    map_primary.png            mappa del sistema con la minima energia primaria
    map_co2.csv, map_primary.csv
    advantage_co2.png          P(il migliore sostenibile batte il cherosene)
    advantage_primary.png
    advantage_co2.csv, advantage_primary.csv
    energy_mix_effects.csv     tabella per missione
    energy_mix_co2.png         barre per missione, CO2
    energy_mix_primary.png     barre per missione, energia primaria
"""
import hashlib
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import (AircraftSizer, ESAFCombustion, EnergyCarrier, Mission,
                  compute_intensity)
from cnav.uncertainty import (CopulaPair, CorrelationModel, ParameterSpec,
                              PropagationResult, TriangularWithFloor,
                              classify_regions, plot_probability_map,
                              probability_map, robust_area_fraction,
                              sample_literature_parameters, theta_row_to_tech_wtt)
from cnav.uncertainty.technology_map import BASE_COLORS

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR.parent / "07_propagazione" / "propagation_result.npz"
KEROSENE_CACHE = OUT_DIR / "kerosene_cube.npz"


# =====================================================================
# 1. CONFIGURAZIONE
# =====================================================================

N_CAMPIONI = None      # quante righe di theta usare: None = tutte.
                       # Un numero piccolo (20-50) serve per provare il
                       # flusso senza aspettare la propagazione intera.
                       # Si prendono le PRIME righe, che in un disegno LHS
                       # non sono speciali rispetto alle altre.
N_WORKERS = 1          # processi per la propagazione del cherosene
CHUNK_SIZE = 25        # campioni per blocco: la cache si aggiorna a fine blocco
SEED = 13              # seed dei parametri di mix, diverso da quelli
                       # delle Fasi 7 e 11: i campioni sono indipendenti
THRESHOLD = 0.90       # soglia di robustezza delle mappe
QUANTILI = (0.05, 0.95)   # estremi degli intervalli riportati e disegnati
CHECK_REGRESSIONE = True
FONDO_PDF = 0.18       # peso dell'uniforme nelle PDF dei parametri a monte:
                       # è quanto le distribuzioni vengono sollevate da zero
                       # agli estremi. Vedi TriangularWithFloor in
                       # distributions.py
RHO_ETA_G = -0.7       # dentro una catena: rendimento ed emissione si muovono
                       # in verso opposto. Stesso valore dello scenario
                       # attuale di supply_chain_maps.py
QUOTA_ELETTRICA_RAFFINAZIONE = 0.04   # frazione delle emissioni a
                       # monte del cherosene che segue la rete. Si prende da GREET)

PROPULSORI = ("fan", "propeller")   # stesso ordine della Fase 7
KEROSENE_LABEL = "Kerosene combustion"

# colore del cherosene nelle mappe: BASE_COLORS conosce solo le quattro
# tecnologie sostenibili, e _label_color solleverebbe KeyError. Il grigio
# e' lo stesso usato per il convenzionale nei grafici a barre
BASE_COLORS.setdefault(KEROSENE_LABEL, "#8C8C8C")


def intorno(moda: float, frazione: float = 0.05) -> tuple:
    """Terna (minimo, moda, massimo) simmetrica: moda +/- frazione*moda."""
    return (moda * (1 - frazione), moda, moda * (1 + frazione))


# ---------------------------------------------------------------------
# L'energy mix, riassunto in due parametri incerti.
#
# Entrambi sono già valutati sull'insieme di tutte le fonti primarie, e
# si dichiarano come terna (minimo, moda, massimo) di una triangolare.
# La terna puo' essere asimmetrica: intorno() e' solo la scorciatoia per
# il caso simmetrico.
#
#   ETA_CATENA        rendimento medio di catena dalla fonte primaria
#                     all'elettricità sulla rete: MJ elettrici prodotti
#                     per MJ di energia primaria impiegata. L'energia
#                     primaria richiesta è l'elettricità divisa per
#                     questo rendimento.
#
#   GCO2_PER_MJ_EL    intensità di carbonio dell'elettricità: grammi di
#                     CO2 per ogni MJ elettrico prodotto. Se il dato di
#                     partenza è in gCO2/kWh, dividerlo per 3.6.
# ---------------------------------------------------------------------
ETA_CATENA = intorno(0.457875458)
GCO2_PER_MJ_EL = intorno(140.26)
FONTE_MIX = "GREET"

# ---------------------------------------------------------------------
# Il riferimento convenzionale a cherosene (Jet A-1).
#
#   KEROSENE_LHV_J_PER_KG          potere calorifico inferiore
#   KEROSENE_ETA_WELL_TO_TANK      MJ di cherosene al serbatoio per MJ di
#                                  greggio estratto (estrazione,
#                                  raffinazione, trasporto). Incerto.
#   KEROSENE_GCO2_PER_MJ_BURN      CO2 della combustione a bordo, per MJ
#                                  bruciato (3160 g per kg diviso 43 MJ/kg
#                                  da circa 73.5). NON incerto: è
#                                  stechiometria, non una stima di filiera.
#   KEROSENE_GCO2_PER_MJ_UPSTREAM  CO2 della filiera a monte, per MJ di
#                                  cherosene al serbatoio. Incerto.
#
# L'intensità totale del cherosene è la somma delle due voci: la parte
# certa più quella campionata.
# ---------------------------------------------------------------------
KEROSENE_LHV_J_PER_KG = 43.0e6
KEROSENE_ETA_WELL_TO_TANK = intorno(0.900455)
KEROSENE_GCO2_PER_MJ_BURN = 73.5
KEROSENE_GCO2_PER_MJ_UPSTREAM = intorno(7.402)
FONTE_KEROSENE = "ICAO e GREET"

# ---------------------------------------------------------------------
# Le missioni della tabella e dei grafici a barre.
#
# Scelte sulla mappa best-system in modo da diversificare il vincitore:
# i primi punti stanno nel dominio della batteria, quelli centrali in
# quello della fuel cell, gli ultimi in quello della combustione a
# idrogeno. Non si valuta il modello qui: ogni missione viene agganciata
# al nodo di griglia più vicino del cubo, e lo script stampa il nodo
# effettivamente usato, che in generale non coincide con i valori scritti
# nell'etichetta.
# ---------------------------------------------------------------------
MISSIONI = [
    ("10 nmi, 200 kt (propeller)",    Mission(range_nmi=10.0,    cruise_speed_kt=200.0, propulsor="propeller")),
    ("50 nmi, 250 kt (propeller)",    Mission(range_nmi=50.0,    cruise_speed_kt=250.0, propulsor="propeller")),
    ("300 nmi, 250 kt (propeller)",   Mission(range_nmi=300.0,   cruise_speed_kt=250.0, propulsor="propeller")),
    ("1500 nmi, 250 kt (propeller)",  Mission(range_nmi=1500.0,  cruise_speed_kt=250.0, propulsor="propeller")),
    ("100 nmi, 450 kt (fan)",         Mission(range_nmi=100.0,   cruise_speed_kt=450.0, propulsor="fan")),
    ("1500 nmi, 450 kt (fan)",        Mission(range_nmi=1500.0,  cruise_speed_kt=450.0, propulsor="fan")),
    ("6000 nmi, 450 kt (fan)",        Mission(range_nmi=6000.0,  cruise_speed_kt=450.0, propulsor="fan")),
    ("10000 nmi, 450 kt (fan)",       Mission(range_nmi=10000.0, cruise_speed_kt=450.0, propulsor="fan")),
]

NOME_BREVE = {
    "Battery-electric": "batteria",
    "Hydrogen fuel cell": "fuel cell H2",
    "Hydrogen combustion": "combustione H2",
    "e-SAF combustion": "e-SAF",
    KEROSENE_LABEL: "cherosene",
}


# =====================================================================
# 2. IL CHEROSENE SULLA GRIGLIA
# =====================================================================

class KeroseneCombustion(ESAFCombustion):
    """Il velivolo convenzionale a cherosene.

    Nel tank-to-wake è identico all'e-SAF (stessa cellula, stesso motore,
    stesso OEW: l'e-SAF è un drop-in), e differisce solo per il potere
    calorifico e per cosa c'è a monte del serbatoio. Il carrier ha
    efficienza 1.0 perché la catena a monte del cherosene non passa per
    l'elettricità e viene applicata a valle, sull'energia al serbatoio.
    """

    name = KEROSENE_LABEL

    def __init__(self, lhv_J_per_kg: float = KEROSENE_LHV_J_PER_KG):
        super().__init__(EnergyCarrier("Jet A-1 (fossile)", 1.0))
        self.lhv_J_per_kg = lhv_J_per_kg

    def specific_energy_J_per_kg(self, tech) -> float:
        return self.lhv_J_per_kg


def etichette_cherosene() -> list:
    """Le due etichette del cherosene, nell'ordine dei propulsori della Fase 7."""
    return [f"{KEROSENE_LABEL} ({p})" for p in PROPULSORI]


def valuta_cherosene(row, grid) -> np.ndarray:
    """Energia di combustibile al serbatoio per pax*nmi, (2, n_speeds, n_ranges).

    Stessa convenzione della Fase 7: NaN dove il dimensionamento non
    converge, e i NaN non vanno sostituiti con numeri grandi.

    Attenzione: qui NON si usa l'intensity, che per il cherosene sarebbe
    l'energia divisa per l'efficienza fittizia 1.0 del suo carrier e non
    significherebbe nulla. Si usa l'energia effettivamente imbarcata, che
    è il punto di partenza della catena fossile.
    """
    tech, _ = theta_row_to_tech_wtt(row)   # il wtt non serve: il cherosene
                                           # non passa per l'elettricità
    sizer = AircraftSizer(tech)
    sistema = KeroseneCombustion()

    n_speeds, n_ranges = grid.shape
    out = np.full((len(PROPULSORI), n_speeds, n_ranges), np.nan, dtype=np.float32)

    for i, speed_kt in enumerate(grid.speeds_kt):
        for j, range_nmi in enumerate(grid.ranges_nmi):
            for p_idx, propulsore in enumerate(PROPULSORI):
                mission = Mission(range_nmi=float(range_nmi),
                                  cruise_speed_kt=float(speed_kt),
                                  propulsor=propulsore)
                try:
                    res = compute_intensity(mission, sistema, tech, sizer)
                    if res.sizing.converged and res.n_pax > 0:
                        out[p_idx, i, j] = (res.energy_at_tank_MJ
                                            / (res.n_pax * mission.range_nmi))
                except Exception:
                    # un theta patologico non deve far cadere il run:
                    # resta NaN, cioe' "non fattibile"
                    pass
    return out


# --- worker per la parallelizzazione ---------------------------------
_WORKER_GRID = None


def _worker_init(grid):
    global _WORKER_GRID
    _WORKER_GRID = grid


def _worker_eval(args):
    idx, row = args
    return idx, valuta_cherosene(row, _WORKER_GRID)


def _firma(theta_sel: pd.DataFrame, grid) -> dict:
    """La firma del calcolo del cherosene: theta usata, griglia, LHV.

    Serve a impedire che una cache prodotta con un altro theta o un'altra
    griglia venga riusata in silenzio: i campioni sono identificati dal
    solo indice di riga, quindi senza questo controllo si mescolerebbero
    run diversi. Stessa logica del marker dei checkpoint della Fase 7.
    """
    valori = np.ascontiguousarray(theta_sel.to_numpy(dtype=float)).tobytes()
    return {
        "theta_sha1": hashlib.sha1(valori).hexdigest()[:16],
        "griglia": f"{len(grid.speeds_kt)}x{len(grid.ranges_nmi)}",
        "lhv": f"{KEROSENE_LHV_J_PER_KG:.0f}",
        "n_campioni": f"{len(theta_sel)}",
    }


def _leggi_cache(firma: dict, forma: tuple):
    """(cubo, fatti) dalla cache se la firma coincide, altrimenti (None, None)."""
    if not KEROSENE_CACHE.exists():
        return None, None
    dati = np.load(KEROSENE_CACHE, allow_pickle=True)
    precedente = {k: str(dati[f"firma_{k}"]) for k in firma if f"firma_{k}" in dati}
    diverse = [k for k, v in firma.items() if precedente.get(k) != v]
    if diverse:
        print(f"  [!] {KEROSENE_CACHE.name} e' di un altro run "
              f"(diverso: {', '.join(diverse)}): viene ricalcolato da zero")
        return None, None
    cubo = dati["energia_al_serbatoio"]
    fatti = dati["fatti"]
    if cubo.shape != forma:
        return None, None
    return cubo, fatti


def _scrivi_cache(cubo: np.ndarray, fatti: np.ndarray, firma: dict) -> None:
    np.savez_compressed(KEROSENE_CACHE, energia_al_serbatoio=cubo, fatti=fatti,
                        labels=np.array(etichette_cherosene(), dtype=object),
                        **{f"firma_{k}": v for k, v in firma.items()})


def cubo_cherosene(theta_sel: pd.DataFrame, grid) -> np.ndarray:
    """Propaga il cherosene sulle righe di theta_sel, con cache e ripresa.

    Le righe sono le stesse della Fase 7: il cherosene condivide cellula
    e rendimenti con l'e-SAF, quindi il confronto deve restare accoppiato
    campione per campione.
    """
    n = len(theta_sel)
    n_speeds, n_ranges = grid.shape
    forma = (n, len(PROPULSORI), n_speeds, n_ranges)
    firma = _firma(theta_sel, grid)

    cubo, fatti = _leggi_cache(firma, forma)
    if cubo is None:
        cubo = np.full(forma, np.nan, dtype=np.float32)
        fatti = np.zeros(n, dtype=bool)
    else:
        cubo = np.array(cubo)   # copia scrivibile
        fatti = np.array(fatti)
        if fatti.all():
            print(f"  Cherosene: tutti i {n} campioni ripresi da "
                  f"{KEROSENE_CACHE.name}, NON ricalcolati")
            return cubo
        print(f"  Cherosene: {int(fatti.sum())}/{n} campioni ripresi da "
              f"{KEROSENE_CACHE.name}")

    da_fare = np.flatnonzero(~fatti)
    print(f"  Cherosene: {len(da_fare)} campioni da propagare "
          f"({2 * n_speeds * n_ranges} valutazioni ciascuno, {N_WORKERS} worker)")

    t0 = time.time()
    fatti_sessione = 0
    for inizio in range(0, len(da_fare), CHUNK_SIZE):
        blocco = da_fare[inizio:inizio + CHUNK_SIZE]
        payload = [(int(i), theta_sel.iloc[i].to_dict()) for i in blocco]

        if N_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=N_WORKERS,
                                     initializer=_worker_init,
                                     initargs=(grid,)) as pool:
                risultati = list(pool.map(_worker_eval, payload))
        else:
            risultati = [(i, valuta_cherosene(row, grid)) for i, row in payload]

        for i, valori in risultati:
            cubo[i] = valori
            fatti[i] = True
        fatti_sessione += len(risultati)

        _scrivi_cache(cubo, fatti, firma)   # la cache e' anche il checkpoint

        trascorso = time.time() - t0
        ritmo = trascorso / fatti_sessione
        eta_min = ritmo * (len(da_fare) - fatti_sessione) / 60.0
        print(f"    [{int(fatti.sum())}/{n}] {trascorso:.0f}s, "
              f"~{ritmo:.2f}s/campione, ETA ~{eta_min:.0f} min")

    return cubo


# =====================================================================
# 3. LE PDF DEI PARAMETRI A MONTE
# =====================================================================

def _valida_terna(terna, dove: str, limiti: tuple = None) -> tuple:
    """Controlla una terna (minimo, moda, massimo) e la restituisce in float.

    I controlli su ordine e degenerazione ripetono quelli di triangular(),
    ma qui il messaggio dice quale parametro è sbagliato.
    limiti = (lo, hi) impone lo < minimo e massimo <= hi.
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
        if a <= lo or b > hi:
            raise ValueError(f"{dove}: i valori devono stare in ({lo}, {hi}], "
                             f"trovato {terna}")
    return a, m, b


def _spec(nome: str, terna, fonte: str, motivazione: str,
          limiti: tuple = None) -> ParameterSpec:
    a, m, b = _valida_terna(terna, nome, limiti)
    return ParameterSpec(
        name=nome,
        dist=TriangularWithFloor(a, m, b, FONDO_PDF),
        nominal=m,
        pdf_label=f"Triangolare({a:.6g}, {m:.6g}, {b:.6g}) + fondo {FONDO_PDF:.2f}",
        source=fonte,
        rationale=motivazione,
        mode=m,
    )


def build_upstream_specs() -> dict:
    """Le PDF dei quattro parametri a monte, nel formato della Fase 5.

    Tenute fuori da build_default_specs() per lo stesso motivo dei
    parametri di filiera della Fase 11: aggiungerle lì cambierebbe la
    dimensione di theta e il disegno LHS, e i risultati delle Fasi 7, 9 e
    10 non sarebbero più riproducibili.
    """
    specs = [
        _spec("eta_catena", ETA_CATENA, FONTE_MIX,
              "Rendimento medio di catena dalla fonte primaria all'elettricità di rete",
              limiti=(0.0, 1.0)),
        _spec("gco2_el", GCO2_PER_MJ_EL, FONTE_MIX,
              "Intensità di carbonio dell'elettricità di rete, per MJ elettrico"),
        _spec("eta_kerosene", KEROSENE_ETA_WELL_TO_TANK, FONTE_KEROSENE,
              "Rendimento well-to-tank del cherosene: estrazione, raffinazione, trasporto",
              limiti=(0.0, 1.0)),
        _spec("gco2_kerosene_upstream", KEROSENE_GCO2_PER_MJ_UPSTREAM, FONTE_KEROSENE,
              "CO2 della filiera a monte del cherosene, per MJ al serbatoio "
              f"(la combustione, {KEROSENE_GCO2_PER_MJ_BURN:.1f} g/MJ, è deterministica)"),
    ]
    return {s.name: s for s in specs}


def build_upstream_correlations() -> CorrelationModel:
    """I legami fra i quattro parametri a monte.

    Il driver è gco2_el, cioè quanto è pulita la rete: da lì scendono il
    rendimento della rete stessa e, con peso pari alla quota elettrica
    della raffinazione, le emissioni a monte del cherosene. Le copule
    agiscono sui quantili prima della trasformata inversa, quindi le
    marginali restano esattamente quelle dichiarate.

    Nota su cosa cambia davvero i numeri: in questo script ogni metrica
    dipende da un parametro solo, la CO2 dalle due intensità e l'energia
    primaria dai due rendimenti. L'unico legame interno a una stessa
    metrica è quindi quello fra rete e cherosene, ed è l'unico che sposta
    le mappe. Gli altri due ci sono per coerenza con
    supply_chain_maps.py, che usa gli stessi numeri per la filiera
    'grid': dichiararli dipendenti là e indipendenti qui non si
    sosterrebbe in tesi.
    """
    return CorrelationModel(copulas=(
        CopulaPair(a="gco2_el", b="eta_catena", rho=RHO_ETA_G,
                   rationale="Rete: mix più pulito, rendimento di catena più alto"),
        CopulaPair(a="gco2_el", b="gco2_kerosene_upstream",
                   rho=QUOTA_ELETTRICA_RAFFINAZIONE,
                   rationale="La raffinazione consuma elettricità di rete"),
        CopulaPair(a="gco2_kerosene_upstream", b="eta_kerosene", rho=RHO_ETA_G,
                   rationale="Cherosene: rendimento ed emissione in verso opposto"),
    ))


def sample_upstream(n_samples: int, specs: dict, seed: int = SEED) -> pd.DataFrame:
    """Campiona i parametri a monte con lo stesso motore LHS della Fase 5.

    Le dipendenze sono dichiarate in build_upstream_correlations e
    agiscono sui quantili, quindi le marginali restano quelle di
    build_upstream_specs.
    """
    return sample_literature_parameters(n_samples, specs=specs,
                                        correlations=build_upstream_correlations(),
                                        seed=seed)


def stampa_correlazioni(theta_up: pd.DataFrame,
                        correlations: CorrelationModel) -> None:
    """Correlazioni realizzate nel campione contro quelle volute.

    Stesso controllo della Fase 7: quello che si voleva imporre va
    verificato, non dato per fatto.
    """
    print("correlazioni realizzate (target fra parentesi):")
    for pair in correlations.copulas:
        r = theta_up[pair.a].corr(theta_up[pair.b])
        print(f"  {pair.a:<22} / {pair.b:<22} r = {r:+.3f}  ({pair.rho:+.2f})")


# =====================================================================
# 4. DAI DUE CUBI ALLE DUE METRICHE
# =====================================================================

def base_e_etichette(result: PropagationResult, kero: np.ndarray, n: int) -> tuple:
    """Concatena il cubo sostenibile e quello del cherosene.

    Le due metà hanno significato diverso e vanno trattate diversamente
    a valle:
      - per i sistemi sostenibili il valore è l'ELETTRICITA' richiesta
        alla fonte, a cui si applica il mix;
      - per il cherosene è l'ENERGIA DI COMBUSTIBILE al serbatoio, a cui
        si applica la catena fossile.
    """
    base = np.concatenate([
        np.asarray(result.intensities[:n], dtype=np.float32),
        np.asarray(kero, dtype=np.float32),
    ], axis=1)
    return base, list(result.labels) + etichette_cherosene()


def cubo_metrica(base: np.ndarray, labels: list, theta_up: pd.DataFrame,
                 metrica: str) -> np.ndarray:
    """Applica mix e catena fossile e restituisce il cubo della metrica.

    metrica: "co2" [g/(pax*nmi)] oppure "primaria" [MJ/(pax*nmi)].
    Un cubo per volta: tenerne due insieme costerebbe il doppio della
    memoria senza servire a niente.
    """
    eta_el = theta_up["eta_catena"].to_numpy(dtype=np.float32)[:, None, None]
    g_el = theta_up["gco2_el"].to_numpy(dtype=np.float32)[:, None, None]
    eta_k = theta_up["eta_kerosene"].to_numpy(dtype=np.float32)[:, None, None]
    g_k = (theta_up["gco2_kerosene_upstream"].to_numpy(dtype=np.float32)[:, None, None]
           + np.float32(KEROSENE_GCO2_PER_MJ_BURN))

    out = np.empty_like(base)
    for j, label in enumerate(labels):
        fossile = label.startswith(KEROSENE_LABEL)
        if metrica == "co2":
            fattore = g_k if fossile else g_el
            out[:, j, :, :] = base[:, j, :, :] * fattore
        elif metrica == "primaria":
            divisore = eta_k if fossile else eta_el
            out[:, j, :, :] = base[:, j, :, :] / divisore
        else:
            raise ValueError(f"metrica sconosciuta: {metrica!r}")
    return out


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


def risultato_con_metrica(result: PropagationResult, cubo: np.ndarray,
                          labels: list, n: int) -> PropagationResult:
    """Impacchetta un cubo di metrica in un PropagationResult.

    Il passaggio che rende gratuito tutto il resto: probability_map,
    classify_regions, robust_area_fraction e plot_probability_map leggono
    solo best_index, labels e grid, quindi funzionano su questa metrica
    esattamente come sull'electricity intensity.
    """
    return PropagationResult(
        intensities=cubo,
        best_index=_best_index(cubo),
        theta=result.theta.iloc[:n],
        grid=result.grid,
        labels=list(labels),
    )


def probabilita_vantaggio(cubo: np.ndarray, labels: list) -> np.ndarray:
    """P(il migliore sostenibile batte il miglior cherosene), punto per punto.

    Il confronto è fatto campione per campione, quindi tiene conto della
    correlazione fra i due: la cellula del cherosene è quella dell'e-SAF
    e condivide le stesse realizzazioni di L/D, OEW e rendimenti.

    I campioni in cui manca uno dei due termini (nessun sistema
    sostenibile fattibile, oppure cherosene non convergente) sono esclusi
    dal conteggio: dove non ne resta nessuno il valore è NaN.
    """
    fossile = np.array([l.startswith(KEROSENE_LABEL) for l in labels])
    with warnings.catch_warnings():
        # le colonne tutte-NaN sono il caso "mai fattibile": voluto
        warnings.simplefilter("ignore", category=RuntimeWarning)
        migliore_s = np.nanmin(cubo[:, ~fossile], axis=1)
        migliore_k = np.nanmin(cubo[:, fossile], axis=1)

    valide = np.isfinite(migliore_s) & np.isfinite(migliore_k)
    conta = valide.sum(axis=0)
    vince = (valide & (migliore_s < migliore_k)).sum(axis=0)
    return np.where(conta > 0, vince / np.maximum(conta, 1), np.nan)


# =====================================================================
# 5. MAPPE
# =====================================================================

def disegna_mappa(res: PropagationResult, titolo: str, slug: str) -> None:
    pmap = probability_map(res)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_probability_map(pmap, threshold=THRESHOLD, ax=ax,
                         title=f"{titolo} (N = {pmap.n_samples})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"map_{slug}.png", dpi=150)
    plt.close(fig)

    pmap.to_dataframe().to_csv(OUT_DIR / f"map_{slug}.csv", index=False)

    regioni = classify_regions(pmap, threshold=THRESHOLD)
    print(f"\n{titolo}")
    print(f"  area robusta (P >= {THRESHOLD:.2f}): "
          f"{100 * float(np.mean(regioni >= 0)):.1f}% del piano")
    for j, label in enumerate(pmap.labels):
        frazione = float(np.mean(regioni == j))
        if frazione > 0:
            print(f"    {label:<32}{100 * frazione:>6.1f}%")
    print("  ripartizione al variare della soglia:")
    print(robust_area_fraction(pmap).round(3).to_string(index=False))


def disegna_vantaggio(prob: np.ndarray, grid, titolo: str, slug: str) -> None:
    """La mappa della probabilità di battere il cherosene.

    Qui il colore è una probabilità sola, non una tecnologia: è la
    domanda "conviene davvero passare al sostenibile, in questo punto del
    piano?", che la mappa del vincitore non risponde.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    mesh = ax.pcolormesh(grid.ranges_nmi, grid.speeds_kt, prob,
                         cmap="RdYlGn", vmin=0.0, vmax=1.0, shading="auto")
    livelli = [0.1, 0.5, 0.9]
    contorni = ax.contour(grid.ranges_nmi, grid.speeds_kt, prob,
                          levels=livelli, colors="k", linewidths=0.9)
    ax.clabel(contorni, inline=True, fontsize=7, fmt="%.1f")
    ax.set_xscale("log")
    ax.set_xlabel("Range [nmi]")
    ax.set_ylabel("Velocità di crociera [kt]")
    ax.set_title(titolo)
    fig.colorbar(mesh, ax=ax, label="probabilità")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"advantage_{slug}.png", dpi=150)
    plt.close(fig)

    righe = [{"range_nmi": r, "speed_kt": v, "P": prob[i, k]}
             for i, v in enumerate(grid.speeds_kt)
             for k, r in enumerate(grid.ranges_nmi)]
    pd.DataFrame(righe).to_csv(OUT_DIR / f"advantage_{slug}.csv", index=False)

    finite = np.isfinite(prob)
    print(f"\n{titolo}")
    print(f"  P >= 0.90 nel {100 * float(np.mean(prob[finite] >= 0.90)):.1f}% "
          f"del piano, P <= 0.10 nel "
          f"{100 * float(np.mean(prob[finite] <= 0.10)):.1f}%")


def check_regressione(res_metrica: PropagationResult, labels: list,
                      best_originale: np.ndarray, nome: str) -> None:
    """Test di non regressione, escludendo il cherosene.

    Fra i soli sistemi sostenibili la metrica è l'electricity intensity
    moltiplicata per una costante uguale per tutte le etichette (g_el, o
    1/eta_catena), quindi il vincitore non può cambiare e deve coincidere
    con quello del cubo della Fase 7. Se questo controllo fallisce
    l'errore è nell'algebra, non nei dati a monte.
    """
    fossile = np.array([l.startswith(KEROSENE_LABEL) for l in labels])
    solo_sostenibili = res_metrica.intensities[:, ~fossile]
    uguale = np.array_equal(_best_index(solo_sostenibili), best_originale)
    print(f"\nTest di non regressione [{nome}], senza cherosene il vincitore "
          f"deve restare quello della Fase 7: {'OK' if uguale else 'FALLITO'}")


# =====================================================================
# 6. TABELLA E BARRE PER MISSIONE
# =====================================================================

def nodo_piu_vicino(grid, mission: Mission) -> tuple:
    """(indice velocità, indice range) del nodo di griglia più vicino.

    Il range si cerca in scala logaritmica, coerentemente con la griglia
    geometrica della Fase 7: in lineare un nodo a 10000 nmi sarebbe
    "vicino" a uno a 7000, in log no.
    """
    i = int(np.argmin(np.abs(grid.speeds_kt - mission.cruise_speed_kt)))
    k = int(np.argmin(np.abs(np.log(grid.ranges_nmi) - np.log(mission.range_nmi))))
    return i, k


def righe_missioni(cubo: np.ndarray, labels: list, grid, metrica: str) -> list:
    """Estrae dal cubo, per ogni missione, la distribuzione dei due
    contendenti nel nodo di griglia più vicino.

    Il vincitore sostenibile è cercato campione per campione fra le
    etichette dello stesso propulsore della missione, quindi può cambiare
    da un campione all'altro: si riporta il vincitore modale e con che
    frequenza vince.
    """
    righe = []
    for ordine, (etichetta, mission) in enumerate(MISSIONI):
        i, k = nodo_piu_vicino(grid, mission)
        colonna = f" ({mission.propulsor})"
        j_sost = [j for j, l in enumerate(labels)
                  if l.endswith(colonna) and not l.startswith(KEROSENE_LABEL)]
        j_kero = [j for j, l in enumerate(labels)
                  if l.endswith(colonna) and l.startswith(KEROSENE_LABEL)]

        valori_s = cubo[:, j_sost, i, k]          # (N, n_sostenibili)
        valori_k = cubo[:, j_kero[0], i, k]       # (N,)

        finito = np.isfinite(valori_s)
        sicuro = np.where(finito, valori_s, np.inf)
        vincitore = np.argmin(sicuro, axis=1)
        migliore = np.take_along_axis(sicuro, vincitore[:, None], axis=1).ravel()
        valide = np.isfinite(migliore) & np.isfinite(valori_k)

        if not valide.any():
            print(f"  [avviso] '{etichetta}': nessun campione fattibile, saltata")
            continue

        conteggi = np.bincount(vincitore[valide], minlength=len(j_sost))
        j_modale = j_sost[int(np.argmax(conteggi))]
        med_s, lo_s, hi_s = _riassunto(migliore[valide])
        med_k, lo_k, hi_k = _riassunto(valori_k[valide])

        righe.append({
            "ordine": ordine,
            "missione": etichetta,
            "metrica": metrica,
            "range_nodo_nmi": float(grid.ranges_nmi[k]),
            "speed_nodo_kt": float(grid.speeds_kt[i]),
            "propulsore": mission.propulsor,
            "vincitore_modale": labels[j_modale],
            "frequenza_vincitore": float(conteggi.max() / conteggi.sum()),
            "sost_mediana": med_s, "sost_lo": lo_s, "sost_hi": hi_s,
            "kero_mediana": med_k, "kero_lo": lo_k, "kero_hi": hi_k,
            "p_sost_meglio": float(np.mean(migliore[valide] < valori_k[valide])),
            "campioni_validi": int(valide.sum()),
        })
    return righe


def _riassunto(campioni: np.ndarray) -> tuple:
    """(mediana, quantile basso, quantile alto)."""
    lo, hi = np.quantile(campioni, QUANTILI)
    return float(np.median(campioni)), float(lo), float(hi)


def stampa_missioni(df: pd.DataFrame) -> None:
    q_lo, q_hi = (int(100 * q) for q in QUANTILI)
    unita = {"co2": "g/(pax*nmi)", "primaria": "MJ/(pax*nmi)"}
    print("\n" + "=" * 78)
    print(f"RISULTATI PER MISSIONE  (mediana e intervallo {q_lo}-{q_hi}%)")
    print("=" * 78)
    for (missione,), blocco in df.groupby(["missione"], sort=False):
        testa = blocco.iloc[0]
        print(f"\n{missione}")
        print(f"  nodo di griglia usato: {testa.range_nodo_nmi:.0f} nmi, "
              f"{testa.speed_nodo_kt:.0f} kt   "
              f"(campioni validi: {testa.campioni_validi})")
        for _, riga in blocco.iterrows():
            print(f"  [{riga.metrica}]  {unita[riga.metrica]}")
            print(f"    migliore sostenibile  {riga.sost_mediana:>10.3f} "
                  f"[{riga.sost_lo:.3f}, {riga.sost_hi:.3f}]   "
                  f"{riga.vincitore_modale} nel {100 * riga.frequenza_vincitore:.0f}% dei campioni")
            print(f"    cherosene             {riga.kero_mediana:>10.3f} "
                  f"[{riga.kero_lo:.3f}, {riga.kero_hi:.3f}]")
            print(f"    P(sostenibile meglio del cherosene) = "
                  f"{100 * riga.p_sost_meglio:.1f}%")


def plot_missioni(df: pd.DataFrame, metrica: str, ylabel: str, titolo: str,
                  filename: str) -> Path:
    """Barre affiancate, migliore sostenibile e cherosene, con barra d'errore.

    L'altezza è la mediana, la barra d'errore copre l'intervallo fra i
    quantili di QUANTILI ed è asimmetrica. Sopra la barra sostenibile è
    scritto quale sistema vince quella missione nella maggioranza dei
    campioni, che altrimenti la figura da sola non direbbe.
    """
    blocco = df[df.metrica == metrica]
    etichette = list(blocco.missione)
    x = np.arange(len(etichette))
    larghezza = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    serie = [("sost", "migliore sostenibile", "#4C72B0"),
             ("kero", "cherosene", "#8C8C8C")]
    altezza_max = 0.0
    for p, (prefisso, nome, colore) in enumerate(serie):
        mediane = blocco[f"{prefisso}_mediana"].to_numpy(dtype=float)
        bassi = blocco[f"{prefisso}_lo"].to_numpy(dtype=float)
        alti = blocco[f"{prefisso}_hi"].to_numpy(dtype=float)
        errori = np.vstack([mediane - bassi, alti - mediane])
        posizioni = x + (p - 0.5) * larghezza
        ax.bar(posizioni, mediane, larghezza, label=nome, color=colore,
               yerr=errori, capsize=3, ecolor="#333333", error_kw={"linewidth": 1})
        altezza_max = max(altezza_max, float(np.nanmax(alti)))

        if prefisso == "sost":
            for xi, alto, vincitore in zip(posizioni, alti, blocco.vincitore_modale):
                nome_breve = NOME_BREVE.get(vincitore.rsplit(" (", 1)[0], vincitore)
                ax.annotate(nome_breve, (xi, alto), textcoords="offset points",
                            xytext=(0, 6), ha="center", va="bottom", rotation=90,
                            fontsize=8, color="#4C72B0")

    q_lo, q_hi = (int(100 * q) for q in QUANTILI)
    ax.set_ylim(0, altezza_max * 1.35)   # spazio per le etichette verticali
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{titolo}\nmediana e intervallo {q_lo}-{q_hi}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =====================================================================
# 7. MAIN
# =====================================================================

METRICHE = {
    "co2": {
        "slug": "co2",
        "titolo_mappa": "Sistema con la minima CO2",
        "titolo_vantaggio": "P(il migliore sostenibile emette meno CO2 del cherosene)",
        "ylabel": "CO2 [g/(pax*nmi)]",
        "titolo_barre": "Emissioni di CO2: sostenibile vs convenzionale",
        "file_barre": "energy_mix_co2.png",
    },
    "primaria": {
        "slug": "primary",
        "titolo_mappa": "Sistema con la minima energia primaria",
        "titolo_vantaggio": "P(il migliore sostenibile richiede meno energia primaria del cherosene)",
        "ylabel": "Energia primaria [MJ/(pax*nmi)]",
        "titolo_barre": "Energia primaria richiesta: sostenibile vs convenzionale",
        "file_barre": "energy_mix_primary.png",
    },
}


def main() -> None:
    result = PropagationResult.load(RESULT_PATH)
    n = result.n_samples if N_CAMPIONI is None else min(int(N_CAMPIONI), result.n_samples)
    theta_sel = result.theta.iloc[:n]
    print(f"Cubo della Fase 7: {result.n_samples} campioni su griglia "
          f"{len(result.grid.ranges_nmi)} x {len(result.grid.speeds_kt)}")
    print(f"Campioni usati qui: {n}")

    # --- il cherosene sulla stessa griglia e sulle stesse righe ------
    print("\n" + "=" * 78)
    print("PROPAGAZIONE DEL CHEROSENE")
    print("=" * 78)
    kero = cubo_cherosene(theta_sel, result.grid)

    # --- i parametri a monte -----------------------------------------
    specs = build_upstream_specs()
    theta_up = sample_upstream(n, specs)

    tabella = pd.DataFrame([
        {"parametro": s.name, "nominale": s.nominal, "PDF": s.pdf_label,
         "fonte": s.source, "motivazione": s.rationale}
        for s in specs.values()
    ])
    print("\n" + "=" * 78)
    print("PARAMETRI A MONTE")
    print("=" * 78)
    print(tabella[["parametro", "nominale", "PDF"]].to_string(index=False))
    print(f"(la combustione del cherosene, {KEROSENE_GCO2_PER_MJ_BURN:.1f} g/MJ, "
          f"è deterministica)\n")
    stampa_correlazioni(theta_up, build_upstream_correlations())
    tabella.to_csv(OUT_DIR / "energy_mix_specs.csv", index=False)

    base, labels = base_e_etichette(result, kero, n)
    best_originale = result.best_index[:n]

    # --- una metrica per volta ---------------------------------------
    righe = []
    for metrica, info in METRICHE.items():
        cubo = cubo_metrica(base, labels, theta_up, metrica)
        res_m = risultato_con_metrica(result, cubo, labels, n)

        if CHECK_REGRESSIONE:
            check_regressione(res_m, labels, best_originale, metrica)

        disegna_mappa(res_m, info["titolo_mappa"], info["slug"])
        disegna_vantaggio(probabilita_vantaggio(cubo, labels), result.grid,
                          info["titolo_vantaggio"], info["slug"])
        righe += righe_missioni(cubo, labels, result.grid, metrica)

        del cubo, res_m   # un cubo per volta: sono centinaia di MB

    # --- tabella e barre ---------------------------------------------
    # le righe arrivano raggruppate per metrica: si riordinano per
    # missione, che e' come si leggono in tabella e in figura
    df = (pd.DataFrame(righe)
          .sort_values(["ordine", "metrica"], kind="stable")
          .drop(columns="ordine")
          .reset_index(drop=True))
    stampa_missioni(df)
    df.to_csv(OUT_DIR / "energy_mix_effects.csv", index=False)

    for metrica, info in METRICHE.items():
        plot_missioni(df, metrica, info["ylabel"], info["titolo_barre"],
                      info["file_barre"])

    print(f"\nTutto salvato in: {OUT_DIR}")


if __name__ == "__main__":
    # il guard serve davvero: con N_WORKERS > 1 i processi figli
    # reimportano questo file, e senza guard rieseguirebbero tutto
    main()