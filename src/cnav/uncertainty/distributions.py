"""
Definizione dell'incertezza parametrica e
assemblaggio del campione theta da propagare

Questo modulo contiene TRE parti, nell'ordine in cui servono:

1. la TABELLA delle PDF: per ogni parametro incerto, valore nominale, 
   intervallo, tipo di distribuzione, fonte e motivazione. 
   La tabella che finisce in tesi si genera dal
   codice (specs_table()) e non puo' divergere da cio' che viene
   effettivamente campionato

2. le CORRELAZIONI FISICHE imposte fra alcuni di questi parametri

3. il CAMPIONAMENTO vero e proprio (LHS + trasformata inversa) e la
   fusione con i campioni di calibrazione Theta_acc prodotti 
   (cnav.calibration.calibration_uncertainty), a formare la matrice
   theta completa richiesta per la propagazione

DUE FAMIGLIE DI PARAMETRI, DUE ORIGINI DIVERSE
----------------------------------------------
La matrice theta finale ha due blocchi di colonne che non vanno trattati
allo stesso modo:

  - blocco della "incertezza parametrica" (questo modulo): parametri con un intervallo
    plausibile documentabile da fonti esterne. Hanno una PDF dichiarata
    e si campionano per trasformata inversa da LHS.
  - blocco della "incertezza di calibrazione" (theta_acc.csv): parametri senza PDF
    da letteratura, la cui incertezza è dedotta dalla regione
    accettabile Theta_acc = {theta : J(theta) <= J_thr}. Non si
    ricostruiscono PDF marginali per questi, ma si pescano le rghe intere di
    Theta_acc per non distruggere la struttura di dipendenza (eventuale equifinality) 
    emersa dalla calibrazione.

I due blocchi sono disgiunti per costruzione (assemble_theta lo
verifica e solleva un errore se un parametro compare in entrambi).

AVVERTENZA SUL NOMINALE FUORI DALLA PDF
---------------------------------------
Per alcuni parametri il valore dedotto dal paper non coincide con
quello scelto come nominale della PDF assegnata, e in due casi
(e_battery_Wh_per_kg, fuel_cell_specific_power_kW_per_kg) sta
esattamente sull'estremo ottimistico dell'intervallo.

Quindi, la mappa probabilistica non sarà centrata sulla
mappa deterministica, sarà sistematicamente spostata a sfavore di
batteria e fuel cell. check_nominal_consistency() stampa il percentile del
nominale per ogni parametro, ed e' pensata per essere eseguita e
riportata prima della propagazione, non dopo aver visto i risultati.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from ..model.constants import TechAssumptions, WellToTankEfficiencies

__all__ = [
    "ParameterSpec",
    "triangular",
    "scaled_beta",
    "build_default_specs",
    "specs_table",
    "DerivedParameter",
    "linear_from_endpoints", 
    "induced_spec",
    "CopulaPair",
    "CorrelationModel",
    "build_default_correlations",
    "shared_factor_rho",
    "sample_literature_parameters",
    "load_theta_acc",
    "assemble_theta",
]


# =====================================================================
# 1. Tabella delle PDF (par. 6.1)
# =====================================================================

@dataclass(frozen=True)
class ParameterSpec:
    """Un parametro incerto con PDF dichiarata e la sua documentazione.

    I campi riproducono:
    nominale, intervallo, tipo di distribuzione (implicito in dist e
    riassunto in pdf_label), fonte, motivazione, correlazioni (queste
    ultime non qui ma in CorrelationModel, perchè riguardano coppie di
    parametri e non il singolo).

    name deve essere il nome esatto di un campo di TechAssumptions o di
    WellToTankEfficiencies, così theta_to_tech_wtt sa dove metterlo
    senza bisogno di una tabella di traduzione a parte.
    """
    name: str
    dist: object                # distribuzione scipy già "congelata"
    nominal: float
    pdf_label: str              # es. "Triangolare(135, 225, 300)"
    source: str                 # da dove viene l'intervallo
    rationale: str              # perche' questa forma e non un'altra
    mode: Optional[float] = None  # valore più probabile, dichiarato qui
                                  # esplicitamente da chi scrive la spec.
                                  # None quando non esiste (es. Uniforme)

    def ppf(self, u):
        """Trasformata inversa: da quantile uniforme a valore fisico"""
        return self.dist.ppf(u)

    @property
    def support(self) -> tuple:
        return float(self.dist.ppf(0.0)), float(self.dist.ppf(1.0))


def triangular(a: float, m: float, b: float):
    """Triangolare(min=a, moda=m, max=b), nella parametrizzazione della
    guideline (eq. 11). scipy la vuole invece come (c, loc, scale) con
    c = (m - a) / (b - a)"""
    if not (a <= m <= b) or a >= b:
        raise ValueError(f"Triangolare non valida: a={a}, m={m}, b={b}")
    return stats.triang(c=(m - a) / (b - a), loc=a, scale=b - a)


def scaled_beta(alpha: float, beta: float, a: float, b: float):
    """Beta(alpha, beta) riscalata sull'intervallo [a, b], cioe'
    theta = a + (b - a) * Z con Z ~ Beta(alpha, beta).

    Forma raccomandata per efficienze e frazioni:
    tiene il supporto dentro limiti fisici senza le code infinite di una
    Gaussiana
    """
    return stats.beta(alpha, beta, loc=a, scale=b - a)


def build_default_specs() -> dict:
    """La tabella per gli 11 parametri incerti.

    Ritorna un dict {nome: ParameterSpec} ordinato: l'ordine di
    inserimento è l'ordine delle colonne nella matrice theta, quindi
    resta stabile fra un run e l'altro (importante per riprodurre lo
    stesso campione a parità di seed).

    Le fonti indicate nei campi source sono un segnaposto da sostituire con
    i riferimenti bibliografici veri: il codice non può verificarli
    """
    nom = TechAssumptions()
    nom_wtt = WellToTankEfficiencies()

    specs = [
        # --- batteria -------------------------------------------------
        ParameterSpec(
            name="e_battery_Wh_per_kg",
            dist=triangular(135.0, 225.0, 300.0),
            nominal=nom.e_battery_Wh_per_kg,
            pdf_label="Triangolare(135, 225, 300)",
            mode=225.0,
            source="Letteratura",
            rationale=(
                "intervallo credibile con un valore più plausibile al centro, "
                "ma senza base statistica per una forma piu' informativa "
                "ATTENZIONE: il nominale del modello (300) è l'estremo superiore, "
                "cioè il caso piu' ottimistico dell'intervallo"
            ),
        ),
        # --- fuel cell ------------------------------------------------
        ParameterSpec(
            name="fuel_cell_specific_power_kW_per_kg",
            dist=triangular(1.0, 1.5, 2.0),
            nominal=nom.fuel_cell_specific_power_kW_per_kg,
            pdf_label="Triangolare(1.0, 1.5, 2.0)",
            mode=1.5,
            source="Letteratura",
            rationale=(
                "come sopra: range noto, moda plausibile, nessun dato per una PDF "
                "statistica. Anche qui il nominale (2.0) è l'estremo ottimistico"
            ),
        ),
        ParameterSpec(
            name="eta_fuel_cell",
            dist=triangular(0.45, 0.55, 0.65),
            nominal=nom.eta_fuel_cell,
            pdf_label="Triangolare(0.45, 0.55, 0.65)",
            mode=0.55,
            source="Letteratura",
            rationale=(
                "efficienza limitata in [0,1] ma lontana dai bordi: la triangolare "
                "è sufficiente e più leggibile di una Beta, e il supporto resta "
                "comunque fisicamente ammissibile"
            ),
        ),
        # --- stivaggio idrogeno ---------------------------------------
        ParameterSpec(
            name="gamma_tank",
            dist=triangular(0.47, 0.50, 0.58),
            nominal=nom.gamma_tank,
            pdf_label="Triangolare(0.47, 0.50, 0.58)",
            mode=0.50,
            source="Letteratura",
            rationale=(
                "asimmetrica verso l'alto: il margine di miglioramento tecnologico "
                "è maggiore verso serbatoi più leggeri che verso serbatoi peggiori "
                "di quelli oggi realizzabili. Correlato a "
                "hydrogen_empty_weight_multiplier (vedi CorrelationModel)"
            ),
        ),
        ParameterSpec(
            name="hydrogen_empty_weight_multiplier",
            dist=triangular(1.09, 1.10, 1.12),
            nominal=nom.hydrogen_empty_weight_multiplier,
            pdf_label="Triangolare(1.09, 1.10, 1.12)",
            mode=1.11,
            source="Stime dalla letteratura e modelli semplificati di retrofitting",
            rationale=(
                "Correlato a gamma_tank (vedi CorrelationModel)"
            ),
        ),
        # --- well-to-tank ---------------------------------------------
        ParameterSpec(
            name="electricity",
            dist=stats.uniform(loc=0.83, scale=0.05),
            nominal=nom_wtt.electricity,
            pdf_label="Uniforme(0.83, 0.88)",
            source="Letteratura",
            rationale=(
                "si dispone di un intervallo credibile ma di nessuna base per "
                "considerare alcuni valori più probabili di altri"
            ),
        ),
        ParameterSpec(
            name="liquid_hydrogen",
            dist=triangular(0.42, 0.46, 0.47),
            nominal=nom_wtt.liquid_hydrogen,
            pdf_label="Triangolare(0.42, 0.46, 0.47)",
            mode=0.46,
            source="Letteratura",
            rationale=(
                "Correlata a e_saf tramite l'elettrolisi condivisa"
            ),
        ),
        ParameterSpec(
            name="e_saf",
            dist=triangular(0.21, 0.26, 0.30),
            nominal=nom_wtt.e_saf,
            pdf_label="Triangolare(0.21, 0.26, 0.30)",
            mode=0.26,
            source="Letteratura",
            rationale=(
                "Correlata a liquid_hydrogen tramite l'elettrolisi condivisa"
            ),
        ),
        # --- efficienze di conversione e propulsive -------------------
        ParameterSpec(
            name="eta_motor",
            dist=scaled_beta(3.6, 3.0, 0.85, 0.99),
            nominal=nom.eta_motor,
            pdf_label="0.85 + 0.14 * Beta(3.6, 3.0)",
            mode=0.85,
            source="Letteratura",
            rationale=(
                "efficienza limitata a un intervallo fisico [0.85, 0.99]: Beta "
                "riscalata, leggermente sbilanciata verso l'alto "
                "(alpha > beta) perche' i valori bassi dell'intervallo sono già "
                "superati dalla tecnologia attuale"
            ),
        ),
        ParameterSpec(
            name="eta_p_propeller",
            dist=scaled_beta(4.0, 4.0, 0.80, 0.90),
            nominal=nom.eta_p_propeller,
            pdf_label="0.80 + 0.10 * Beta(4, 4)",
            mode=0.80,
            source="Letteratura",
            rationale=(
                "Beta simmetrica: intervallo fisico noto, nessuna ragione per "
                "preferire una metà"
            ),
        ),
        ParameterSpec(
            name="eta_p_fan",
            dist=scaled_beta(4.0, 4.0, 0.70, 0.80),
            nominal=nom.eta_p_fan,
            pdf_label="0.70 + 0.10 * Beta(4, 4)",
            mode=0.70,
            source="Letteratura",
            rationale="stessa motivazione di eta_p_propeller, su intervallo diverso",
        ),
    ]
    return {s.name: s for s in specs}


# =====================================================================
# 2. Correlazioni fisiche imposte
# =====================================================================


@dataclass(frozen=True)
class CopulaPair:
    """Due parametri con dipendenza forte ma non deterministica, imposta
    con una copula gaussiana di coefficiente rho sulla scala normale.

    Le distribuzioni marginali restano ESATTAMENTE quelle dichiarate in
    ParameterSpec: la copula agisce solo sui quantili uniformi, prima
    della trasformata inversa. E' il modo di introdurre la dipendenza
    senza dover rinunciare alle marginali gia' giustificate al par. 6.1.
    """
    a: str
    b: str
    rho: float
    rationale: str = ""


@dataclass(frozen=True)
class DerivedParameter:
    """Un parametro NON campionato, calcolato da un altro con una
    relazione deterministica lineare:

        theta_follower = intercept + slope * theta_driver

    slope e intercept sono forniti direttamente (non dedotti dagli
    estremi delle due ParameterSpec): la marginale effettiva del
    follower e' quindi quella INDOTTA da questi due coefficienti sulla
    marginale del driver, e va confrontata con induced_spec() rispetto
    a quella eventualmente dichiarata a parte in ParameterSpec.
    """
    name: str
    driver: str
    slope: float
    intercept: float
    rationale: str = ""

    def value(self, driver_values):
        return self.intercept + self.slope * np.asarray(driver_values)

    @property
    def relation_label(self) -> str:
        return f"{self.intercept:.6g} {self.slope:+.6g} * {self.driver}"


def induced_spec(derived: DerivedParameter, driver_spec: ParameterSpec) -> dict:
    """La marginale EFFETTIVA del parametro derivato, cioe' l'immagine
    della marginale del driver attraverso la retta scelta."""
    d_lo, d_hi = driver_spec.support
    ends = sorted([float(derived.value(d_lo)), float(derived.value(d_hi))])
    out = {"min": ends[0], "max": ends[1]}
    params = dict(zip(("c", "loc", "scale"), driver_spec.dist.args))
    params.update(driver_spec.dist.kwds)
    if driver_spec.dist.dist.name == "triang" and {"c", "loc", "scale"} <= set(params):
        mode_driver = params["loc"] + params["c"] * params["scale"]
        out["moda"] = float(derived.value(mode_driver))
    return out


@dataclass(frozen=True)
class CorrelationModel:
    copulas: tuple = ()
    derived: tuple = ()

    def follower_names(self) -> set:
        """Parametri che NON consumano una colonna LHS propria perche'
        ricavati da un altro parametro con una relazione lineare."""
        return {dp.name for dp in self.derived}


def shared_factor_rho(spec_a: ParameterSpec, spec_b: ParameterSpec,
                      shared_dist) -> float:
    """Coefficiente di correlazione IMPLICATO da un fattore comune.

    Situazione: due parametri si scrivono entrambi come prodotto di un
    fattore condiviso e di un fattore proprio indipendente,

        theta_a = F * A ,     theta_b = F * B ,

    con F la quantita' condivisa (qui: l'efficienza di elettrolisi, a
    monte sia della liquefazione dell'idrogeno sia della sintesi
    dell'e-SAF). Passando ai logaritmi la relazione diventa additiva e,
    per fluttuazioni piccole, il coefficiente di correlazione fra
    theta_a e theta_b vale

        rho = sqrt(f_a * f_b),

    dove f_x = Var(log F) / Var(log theta_x) e' la quota di varianza
    (relativa) di theta_x spiegata dal fattore comune. In pratica si usa
    il coefficiente di variazione, CV = sigma / media, come proxy di
    sigma(log).

    Perche' non usare semplicemente una colonna LHS condivisa (rho = 1)?
    Perche' sarebbe incoerente con le marginali gia' fissate: rho = 1
    significa che il fattore comune spiega il 100% della varianza di
    ENTRAMBI, mentre le due catene hanno anche perdite proprie e
    indipendenti (liquefazione e distribuzione da un lato, DAC e sintesi
    dall'altro) che quella dipendenza perfetta cancellerebbe.

    Se per un parametro risulta CV(F) >= CV(theta), la quota f viene
    troncata a 1: vuol dire che l'intervallo assegnato a quel parametro
    e' gia' piu' stretto della sola incertezza del fattore condiviso, il
    che e' un'incoerenza fra le due marginali dichiarate ed e' una cosa
    da segnalare in tesi, non da nascondere.
    """
    cv_shared = float(shared_dist.std()) / float(shared_dist.mean())
    cv_a = float(spec_a.dist.std()) / float(spec_a.dist.mean())
    cv_b = float(spec_b.dist.std()) / float(spec_b.dist.mean())
    f_a = min(1.0, (cv_shared / cv_a) ** 2)
    f_b = min(1.0, (cv_shared / cv_b) ** 2)
    return math.sqrt(f_a * f_b)


def build_default_correlations(specs: Optional[dict] = None,
                               electrolysis_range: tuple = (0.65, 0.72)
                               ) -> CorrelationModel:
    """Le due dipendenze fisiche individuate fra i parametri di letteratura.

    (1) hydrogen_empty_weight_multiplier = 1.139503 - 0.08535*gamma_tank
        Relazione praticamente lineare derivata dall'esercizio di 
        retrofitting di un aereo convenzionale per "abilitarlo" alla 
        combustione dell'idrogeno liquido (retrofit_H2.py)

    (2) liquid_hydrogen <-> e_saf, come COPULA GAUSSIANA con rho
        derivato da shared_factor_rho() sull'efficienza di elettrolisi
        condivisa (electrolysis_range). Non un fattore condiviso, per la
        ragione spiegata in shared_factor_rho: le due catene hanno
        perdite proprie indipendenti a valle dell'elettrolisi.
    """
    specs = specs or build_default_specs()
    lo, hi = electrolysis_range
    electrolysis = stats.uniform(loc=lo, scale=hi - lo)
    rho = shared_factor_rho(specs["liquid_hydrogen"], specs["e_saf"], electrolysis)

    tank_multiplier = DerivedParameter(
    name="hydrogen_empty_weight_multiplier",
    driver="gamma_tank",
    slope=-0.08535,        # <-- il tuo coefficiente angolare
    intercept=1.15420,     # <-- la tua quota
    rationale=(
            "coefficienti stimati; gamma_tank più alto (serbatoio "
            "gravimetricamente migliore) <-> penalità strutturale minore"
        ),
    )

    return CorrelationModel(
        derived=(tank_multiplier,),
        copulas=(
            CopulaPair(
                a="liquid_hydrogen", b="e_saf", rho=rho,
                rationale=(
                    f"elettrolisi condivisa U{electrolysis_range}; rho={rho:.3f} "
                    "derivato dalla quota di varianza spiegata dal fattore comune "
                    "(shared_factor_rho), non assegnato a mano."
                ),
            ),
        ),
    )


def specs_table(specs: Optional[dict] = None,
                correlations: Optional["CorrelationModel"] = None) -> pd.DataFrame:
    """La tabella dei parametri come DataFrame, pronta da esportare in
    tesi (to_latex / to_markdown). Una riga per parametro, colonne
    nominale / range / PDF / fonte / motivazione.

    Se si passa correlations, i parametri DERIVATI (non campionati)
    vengono riportati con la loro relazione e con la marginale INDOTTA
    invece che con la PDF dichiarata a mano in ParameterSpec, che in
    quel caso non e' piu' quella effettivamente usata."""
    specs = specs or build_default_specs()
    derived_by_name = {dp.name: dp for dp in (correlations.derived if correlations else ())}
    rows = []
    for name, s in specs.items():
        if name in derived_by_name:
            dp = derived_by_name[name]
            ind = induced_spec(dp, specs[dp.driver])
            rows.append({
                "parametro": name,
                "nominale": s.nominal,
                "min": ind["min"],
                "max": ind["max"],
                "moda": s.mode,
                "PDF": f"derivato: {dp.relation_label}",
                "fonte": s.source,
                "motivazione": dp.rationale,
            })
            continue
        lo, hi = s.support
        rows.append({
            "parametro": name,
            "nominale": s.nominal,
            "min": lo,
            "max": hi,
            "moda": s.mode,
            "PDF": s.pdf_label,
            "fonte": s.source,
            "motivazione": s.rationale,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 3. Campionamento LHS + fusione con Theta_acc
# =====================================================================

def sample_literature_parameters(n_samples: int,
                                 specs: Optional[dict] = None,
                                 correlations: Optional[CorrelationModel] = None,
                                 seed: int = 0) -> pd.DataFrame:
    """Campiona i parametri di letteratura via LHS + trasformata inversa.

    Procedura:
      a. una matrice LHS di quantili uniformi, una colonna per ogni
         parametro indipendente
      b. si impongono le correlazioni sulla scala uniforme, i follower
         copiano (o ribaltano) la colonna del driver, le coppie con
         copula vengono accoppiate sulla scala normale
      c. trasformata inversa .ppf() colonna per colonna, che porta i
         quantili sui valori fisici seguendo la marginale dichiarata.

    L'ordine b -> c è essenziale: agendo sui quantili e non sui valori,
    le marginali restano esattamente come prima
    """
    specs = specs or build_default_specs()
    correlations = correlations if correlations is not None else build_default_correlations(specs)

    names = list(specs.keys())
    followers = correlations.follower_names()
    unknown = followers - set(names)
    if unknown:
        raise ValueError(f"Follower non presenti fra i parametri: {sorted(unknown)}")

    independent = [nm for nm in names if nm not in followers]
    sampler = qmc.LatinHypercube(d=len(independent), seed=seed)
    u_matrix = sampler.random(n=n_samples)
    u = {nm: u_matrix[:, j] for j, nm in enumerate(independent)}

    # (b) copule gaussiane, sulla scala normale
    for pair in correlations.copulas:
        if pair.a not in u or pair.b not in u:
            raise ValueError(
                f"CopulaPair({pair.a}, {pair.b}): entrambi devono essere parametri "
                "indipendenti (un parametro derivato non puo' entrare in una copula)")
        z_a = stats.norm.ppf(np.clip(u[pair.a], 1e-12, 1 - 1e-12))
        z_b = stats.norm.ppf(np.clip(u[pair.b], 1e-12, 1 - 1e-12))
        z_b_corr = pair.rho * z_a + math.sqrt(1.0 - pair.rho ** 2) * z_b
        u[pair.b] = stats.norm.cdf(z_b_corr)

    # (c) trasformata inversa, solo per i parametri effettivamente campionati
    derived_names = correlations.follower_names()
    data = {nm: specs[nm].ppf(u[nm]) for nm in names if nm not in derived_names}

    # (d) parametri derivati: calcolati dai VALORI FISICI del driver,
    # con la retta scelta (slope, intercept)
    for dp in correlations.derived:
        if dp.driver not in data:
            raise ValueError(f"DerivedParameter {dp.name!r}: driver {dp.driver!r} "
                             "non campionato (non puo' essere a sua volta derivato)")
        data[dp.name] = dp.value(data[dp.driver])

    return pd.DataFrame(data, columns=names)


def load_theta_acc(path: str) -> pd.DataFrame:
    """Legge theta_acc.csv e restituisce solo le
    colonne dei parametri, con i nomi originali.

    Il file prodotto da calibration_uncertainty ha colonne sample_idx,
    cost e theta_<nome>: qui si tengono solo le ultime e si toglie il
    prefisso, così i nomi tornano quelli dei campi di TechAssumptions e
    sono direttamente utilizzabili da theta_to_tech_wtt
    """
    df = pd.read_csv(path)
    theta_cols = [c for c in df.columns if c.startswith("theta_")]
    if not theta_cols:
        raise ValueError(
            f"{path}: nessuna colonna 'theta_<nome>'. Questo file dovrebbe essere "
            "l'output di CalibrationUncertaintyResult.df_accepted")
    out = df[theta_cols].copy()
    out.columns = [c[len("theta_"):] for c in theta_cols]
    return out.reset_index(drop=True)


def _stratified_row_draw(n_draw: int, n_rows: int, seed: int) -> np.ndarray:
    """Estrae n_draw indici di riga da un insieme di n_rows righe, con
    reinserimento ma in modo stratificato: si usa una colonna LHS
    monodimensionale invece di rng.integers, così le righe di
    Theta_acc vengono percorse in modo uniforme anche quando n_draw non
    è molto maggiore di n_rows.

    Serve a evitare che, con un Theta_acc piccolo (poche centinaia di
    righe) e N di propagazione dello stesso ordine, alcune regioni della
    regione accettabile vengano pescate molte volte e altre mai per puro
    effetto del caso
    """
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n_draw)[:, 0]
    return np.minimum((u * n_rows).astype(int), n_rows - 1)


def assemble_theta(n_samples: int,
                   theta_acc: pd.DataFrame,
                   specs: Optional[dict] = None,
                   correlations: Optional[CorrelationModel] = None,
                   seed: int = 0) -> pd.DataFrame:
    """La matrice theta completa da propagare.

    Unisce orizzontalmente:
      - il blocco "letteratura", campionato da PDF dichiarate
        (sample_literature_parameters)
      - il blocco "calibrazione", ottenuto pescando righe intere di
        theta_acc

    I due blocchi sono campionati con seed diversi ma derivati dallo
    stesso seed, e sono indipendenti fra loro, si assume cioè che non
    ci sia dipendenza fra l'incertezza di letteratura e quella di
    calibrazione.

    Ritorna un DataFrame con una riga per campione e una colonna per
    parametro (prima i parametri di letteratura, poi quelli di
    calibrazione)
    """
    specs = specs or build_default_specs()
    lit = sample_literature_parameters(n_samples, specs=specs,
                                       correlations=correlations, seed=seed)

    overlap = set(lit.columns) & set(theta_acc.columns)
    if overlap:
        raise ValueError(
            f"Parametri presenti sia fra quelli di letteratura sia in Theta_acc: "
            f"{sorted(overlap)}. Un parametro deve avere una sola origine di "
            "incertezza: o una PDF di letteratura o la regione accettabile "
            "della calibrazione, non entrambe.")

    idx = _stratified_row_draw(n_samples, len(theta_acc), seed=seed + 1)
    cal = theta_acc.iloc[idx].reset_index(drop=True)

    theta = pd.concat([lit, cal], axis=1)
    theta.index.name = "sample_idx"
    return theta
