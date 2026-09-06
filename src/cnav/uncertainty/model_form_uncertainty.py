"""
Le PDF quando il modello cambia forma: parametri comuni e parametri
specifici.

Il problema
-----------
Cambiando ramo del modello, alcuni parametri incerti smettono di
esistere. Con oew_model="raymer" le sei quantità che descrivono la
regressione del paper (oew_fan_b, oew_fan_c, oew_fan_r_pivot e le tre
gemelle dell'elica) non compaiono in nessuna equazione: campionarle
sarebbe come tirare dei dadi e buttarli via.

Quindi lo spazio dei parametri non è lo stesso fra una variante e
l'altra, e non lo sono nemmeno la sua dimensione e le sue correlazioni
(oew_fan_c è un parametro derivato da altri due: se spariscono i
driver deve sparire anche la relazione).

La soluzione qui adottata
-------------------------
I parametri si dividono in due insiemi:

  COMUNI      presenti sotto ogni forma del modello. Sono la stragrande
              maggioranza: batteria, fuel cell, serbatoi, well-to-tank,
              efficienze. Le loro PDF, e le correlazioni fra loro, non
              cambiano mai
  SPECIFICI   presenti solo sotto certi rami

Perché tenere le PDF comuni identiche fra le varianti: è ciò che rende
il confronto fra le mappe interpretabile. Se cambiassero anche quelle,
una differenza fra due mappe probabilistiche non si saprebbe più
attribuire alla forma del modello o alla diversa descrizione
dell'incertezza dei parametri.

I COEFFICIENTI DELLE VARIANTI SONO DETERMINISTICI
-------------------------------------------------
I numeri che le varianti portano con sé (i coefficienti A e C di
Raymer, il 5% di contingenza, i 30/45 minuti di riserva finale, il
0.75 di velocità di loiter) non sono considerati come parametri incerti: 
sono valori singoli, presi da una tabella di manuale e da un regolamento. 
Stanno in TechAssumptions come tutte le altre costanti del modello, restano
perturbabili dall'analisi di sensibilità locale, ma non hanno una PDF
e non vengono campionati.

Conseguenza importante, da scrivere in tesi e non da nascondere: con
oew_model="raymer" spariscono sei parametri incerti e non ne entra
nessuno. Lo spazio dell'incertezza parametrica passa da 17 a 11
dimensioni, quindi la mappa probabilistica di Raymer sarà più netta
di quella base quasi per costruzione, indipendentemente dai suoi
meriti. Confrontare l'ampiezza delle bande fra le due forme è quindi
un confronto viziato: non si sta misurando quale modello è più certo,
si sta misurando che a uno dei due non è stata attribuita incertezza.
Il confronto che regge è quello sulla posizione dei confini e su quale
tecnologia vince, non sulla loro sfocatura.

Come si confrontano le mappe
----------------------------
Attenzione a un punto che è facile sbagliare. Due propagazioni sotto
forme diverse hanno matrici theta con colonne diverse, quindi non sono
confrontabili campione per campione: il campione 7 dell'una e il
campione 7 dell'altra non descrivono lo stesso velivolo ipotetico. Il
confronto è fra le distribuzioni risultanti, cioè fra le mappe
probabilistiche aggregate, non fra le singole realizzazioni.

Sui parametri comuni si può però usare lo stesso seed, ed è quello che
fa assemble_theta_for_form: riduce il rumore di campionamento nel
confronto, perché la parte condivisa dell'incertezza viene esplorata
in modo simile nelle due mappe. Non le rende appaiate, ma toglie di
mezzo una parte della differenza che sarebbe solo rumore.

La calibrazione
---------------
Theta_acc viene riusato identico sotto ogni forma. È l'assunzione
dichiarata in model_form.py: a rigore ogni variante avrebbe una propria
regione accettabile, perché il peso passeggero che meglio riproduce le
curve del paper dipende anche dal modello dei pesi. Vale la pena
scrivere in tesi che questa è la parte debole del confronto, e che il
modo di chiuderla sarebbe rifare la calibrazione per ogni forma, non
un'analisi in più su quella esistente
"""
from typing import Optional

import pandas as pd

from ..model.model_form import ModelForm
from .distributions import (CorrelationModel, build_default_correlations,
                            build_default_specs, _stratified_row_draw,
                            sample_literature_parameters)

__all__ = [
    "FORM_SPECIFIC_PARAMETERS",
    "common_parameter_names",
    "specs_for_form",
    "correlations_for_specs",
    "assemble_theta_for_form",
    "parameter_membership_table",
]


# =====================================================================
# 1. Chi appartiene a chi
# =====================================================================

# Per ogni (interruttore, valore) i parametri incerti che esistono solo
# quando quel ramo è attivo. Tutto ciò che non compare qui è comune a
# tutte le forme del modello.
#
# Una lista vuota significa: quel ramo cambia le equazioni senza
# introdurre nuova incertezza parametrica. Vale per tutte le varianti
# tranne il ramo "paper" dei pesi, perchp i coefficienti che le altre
# portano con se sono deterministici (vedi il docstring del modulo).
# Si tratta di model-form uncertainty allo stato puro: cambia il modello, non
# la descrizione dell'incertezza.
#
# Se in futuro si decide di attribuire una PDF a uno di quei
# coefficienti, basterà elencarlo qui e dargli un ParameterSpec: il
# resto del modulo non cambia
FORM_SPECIFIC_PARAMETERS = {
    ("oew_model", "paper"): (
        "oew_fan_b", "oew_fan_r_pivot", "oew_fan_c",
        "oew_prop_b", "oew_prop_r_pivot", "oew_prop_c",
    ),
    ("oew_model", "raymer"): (),
    ("aero_model", "fixed"): (),
    ("aero_model", "polar_decay"): (),
    ("reserve_model", "paper"): (),
    ("reserve_model", "easa"): (),
}


def _all_specific_names() -> set:
    return {nm for names in FORM_SPECIFIC_PARAMETERS.values() for nm in names}


def common_parameter_names(specs: Optional[dict] = None) -> list:
    """I parametri incerti presenti sotto ogni forma del modello"""
    specs = specs or build_default_specs()
    specifici = _all_specific_names()
    return [nm for nm in specs if nm not in specifici]


# =====================================================================
# 2. Le tabelle filtrate per forma
# =====================================================================

def specs_for_form(form: ModelForm, specs: Optional[dict] = None) -> dict:
    """I ParameterSpec attivi sotto una data forma del modello.

    L'ordine è quello della tabella completa, quindi i parametri comuni
    mantengono la stessa posizione relativa fra una forma e l'altra. Non
    è un dettaglio estetico: è ciò che permette di riusare lo stesso
    seed sulla parte comune, come fa assemble_theta_for_form
    """
    specs = specs or build_default_specs()
    attivi = set(common_parameter_names(specs))
    for (interruttore, valore), nomi in FORM_SPECIFIC_PARAMETERS.items():
        if getattr(form, interruttore) == valore:
            attivi.update(nomi)

    mancanti = attivi - set(specs)
    if mancanti:
        raise ValueError(
            f"Parametri dichiarati in FORM_SPECIFIC_PARAMETERS ma senza PDF: "
            f"{sorted(mancanti)}. Vanno aggiunti in build_default_specs(), "
            "oppure sono coefficienti deterministici e non vanno elencati "
            "in FORM_SPECIFIC_PARAMETERS")

    return {nm: spec for nm, spec in specs.items() if nm in attivi}


def correlations_for_specs(specs: dict,
                           correlations: Optional[CorrelationModel] = None
                           ) -> CorrelationModel:
    """Le correlazioni ripulite dei riferimenti a parametri assenti.

    Una copula, un derivato o un derivato-da-molti che nomina un
    parametro non presente in specs viene eliminato per intero, non
    riparato: se oew_fan_b sparisce, la relazione che dava oew_fan_c non
    ha piàù senso, e nemmeno oew_fan_c esiste più. Tenerne mezza
    produrrebbe un errore a runtime dentro sample_literature_parameters,
    che è esattamente ciò che questa funzione evita
    """
    correlations = correlations if correlations is not None else build_default_correlations()
    presenti = set(specs)

    copulas = tuple(c for c in correlations.copulas
                    if c.a in presenti and c.b in presenti)
    derived = tuple(d for d in correlations.derived
                    if d.name in presenti and d.driver in presenti)
    derived_many = tuple(d for d in correlations.derived_many
                         if d.name in presenti and all(x in presenti for x in d.drivers))

    return CorrelationModel(copulas=copulas, derived=derived,
                            derived_many=derived_many)


# =====================================================================
# 3. Il campione theta per una forma
# =====================================================================

def assemble_theta_for_form(n_samples: int,
                            theta_acc: pd.DataFrame,
                            form: ModelForm,
                            specs: Optional[dict] = None,
                            seed: int = 0) -> pd.DataFrame:
    """La matrice theta da propagare sotto una data forma del modello.

    Stessa logica di distributions.assemble_theta (blocco letteratura +
    blocco calibrazione), ma con la tabella dei parametri e le
    correlazioni filtrate per la forma.

    I coefficienti deterministici della variante (Raymer, EASA) non
    compaiono in theta: restano al loro valore nominale dentro
    TechAssumptions, dove la forma del modello li va a leggere. Questo è il
    motivo per cui la forma va passata anche a run_propagation, e non
    basta la matrice theta a descrivere il run

    SUL SEED. Passando lo stesso seed a forme diverse, i parametri
    comuni ricevono lo stesso campionamento SOLO se occupano le stesse
    colonne del piano LHS, e non è garantito: se una forma ha meno
    parametri indipendenti, il piano ha meno colonne e i quantili sono
    diversi. Le due mappe restano quindi non appaiate, e il confronto va
    fatto fra distribuzioni (vedi il docstring del modulo). Usare lo
    stesso seed resta comunque preferibile a seed diversi, perchè il
    blocco di calibrazione, che è identico fra le forme, viene pescato
    con le stesse righe
    """
    specs = specs_for_form(form, specs)
    correlations = correlations_for_specs(specs)

    lit = sample_literature_parameters(n_samples, specs=specs,
                                       correlations=correlations, seed=seed)

    overlap = set(lit.columns) & set(theta_acc.columns)
    if overlap:
        raise ValueError(
            f"Parametri presenti sia fra quelli di letteratura sia in Theta_acc: "
            f"{sorted(overlap)}. Un parametro deve avere una sola origine di "
            "incertezza")

    idx = _stratified_row_draw(n_samples, len(theta_acc), seed=seed + 1)
    cal = theta_acc.iloc[idx].reset_index(drop=True)

    theta = pd.concat([lit, cal], axis=1)
    theta.index.name = "sample_idx"
    return theta


# =====================================================================
# 4. La tabella da mettere in tesi
# =====================================================================

def parameter_membership_table(forms: Optional[dict] = None,
                               specs: Optional[dict] = None) -> pd.DataFrame:
    """Quale parametro incerto esiste sotto quale forma del modello.

    Una riga per parametro, una colonna per forma, con "X" dove il
    parametro è attivo. La colonna 'ambito' dice se è comune o
    specifico.

    Sarebbe la tabella che rende leggibile in tesi il fatto che lo spazio
    dell'incertezza cambia dimensione fra le varianti. Va letta insieme
    all'avvertenza del docstring del modulo: le forme con meno righe
    marcate non sono modelli più affidabili, sono modelli ai cui
    coefficienti non è stata attribuita incertezza.

    Non compaiono qui i coefficienti deterministici delle varianti
    (raymer_A_*, reserve_*): non essendo campionati non fanno parte
    dello spazio dell'incertezza. Il posto dove documentarli è la
    tabella delle costanti del modello, insieme a ld_baseline_fan e
    compagnia
    """
    from ..model.model_form import MODEL_FORM_PRESETS

    forms = forms or MODEL_FORM_PRESETS
    specs = specs or build_default_specs()
    comuni = set(common_parameter_names(specs))

    attivi = {nome: set(specs_for_form(f, specs)) for nome, f in forms.items()}

    righe = []
    for nm in specs:
        riga = {"parametro": nm,
                "ambito": "comune" if nm in comuni else "specifico"}
        for nome in forms:
            riga[nome] = "X" if nm in attivi[nome] else ""
        righe.append(riga)

    tab = pd.DataFrame(righe)
    # prima i comuni, poi gli specifici, cosi' la tabella si legge come
    # "questo e' il nocciolo condiviso, questo e' cio' che cambia"
    tab["_ord"] = (tab["ambito"] == "specifico").astype(int)
    return (tab.sort_values(["_ord", "parametro"], kind="stable")
            .drop(columns="_ord").reset_index(drop=True))
