"""
Gli output scalari Y su cui si fa lo screening.

La grandezza èuna sola: la renewable electricity intensity
[MJ/(pax*nmi)], la stessa del modello deterministico

Perchè comunque più di una colonna
------------------------------------
Morris e Sobol hanno bisogno di uno scalare per campione, mentre
l'intensity è una superficie E(R, V) definita su tutto il piano
range-velocità. La superficie non si può dare in pasto agli indici:
la si campiona in un numero finito di MISSIONI RAPPRESENTATIVE,
esattamente come nella sensibilità locale
(OPERATING_CONDITIONS in cnav.sensitivity.local_sensitivity).

Ogni colonna di Y è, quindi, la stessa grandezza fisica, valutata in un
punto diverso della superficie e per un sistema propulsivo diverso. Le
colonne condividono l'unità di misura ma non l'ordine di grandezza (il
fuel cell con fan sta a decine di MJ/pax/nmi, la combustione a idrogeno
a poche unita'), per cui il grafico cumulativo su tutti i punti chiede
comunque una normalizzazione: vedi plots.py

Perchè le missioni non sono le stesse per tutti i sistemi
----------------------------------------------------------
Perchè i domini di fattibilità non lo sono. La batteria non converge
oltre poche decine di miglia, e a 450 kt con fan non converge quasi
mai: chiederle l'intensity a 1500 nmi produce una colonna di soli NaN,
cioè zero effetti elementari e nessuna informazione. Ogni sistema
riceve perciò le missioni della tassonomia dell'analisi di sensibilità locale 
(very short / short a bassa e alta velocità / medium / long), ma troncate al suo 
dominio di fattibilità, più un punto sul propulsore "sbagliato" dove ha senso
(la combustione a elica, il fuel cell con fan) per vedere se il ranking
dei fattori cambia col propulsore.

I NaN
-----
Una configurazione infattibile da NaN, e i NaN non mancano a caso:
mancano dove i parametri sono sfavorevoli. Non si scartano e non si
riempiono, restano NaN e diventano passi non calcolabili negli effetti
elementari (colonna n_ee di morris_indices). Prima di leggere gli
indici si guarda sempre nan_report: sopra il 20-30% di NaN la colonna
dice più sulla fattibilità che sulla sensibilità, e il punto
operativo va spostato più corto o più lento.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..model.aircraft_sizer import AircraftSizer
from ..model.energy_intensity import compute_intensity
from ..model.mission import Mission
from ..model.propulsion_systems import build_default_systems
from ..model.well_to_tank import build_energy_carriers
from ..uncertainty.propagation import theta_row_to_tech_wtt

__all__ = [
    "IntensityAt",
    "MISSION_LIBRARY",
    "REPRESENTATIVE_MISSIONS",
    "default_outputs",
    "outputs_from_missions",
    "outputs_table",
    "evaluate_outputs",
    "nan_report",
]


# ---------------------------------------------------------------------
# La libreria delle missioni
# ---------------------------------------------------------------------
MISSION_LIBRARY = {
    # --- elica ---
    "battery_very_short_low_speed": Mission(range_nmi=10.0,   cruise_speed_kt=200.0, propulsor="propeller"),
    "battery_very_short":           Mission(range_nmi=10.0,   cruise_speed_kt=250.0, propulsor="propeller"),
    "battery_short":                Mission(range_nmi=20.0,   cruise_speed_kt=250.0, propulsor="propeller"),
    "short_low_speed_propeller":    Mission(range_nmi=100.0,  cruise_speed_kt=200.0, propulsor="propeller"),
    "short_high_speed_propeller":   Mission(range_nmi=100.0,  cruise_speed_kt=300.0, propulsor="propeller"),
    "medium_propeller":             Mission(range_nmi=1500.0, cruise_speed_kt=250.0, propulsor="propeller"),
    "long_propeller":               Mission(range_nmi=6000.0, cruise_speed_kt=250.0, propulsor="propeller"),
    # --- fan ---
    "battery_very_short_fan":       Mission(range_nmi=5.0,    cruise_speed_kt=300.0, propulsor="fan"),
    "short_low_speed_fan":          Mission(range_nmi=100.0,  cruise_speed_kt=350.0, propulsor="fan"),
    "short_high_speed_fan":         Mission(range_nmi=100.0,  cruise_speed_kt=450.0, propulsor="fan"),
    "medium_fan":                   Mission(range_nmi=1500.0, cruise_speed_kt=450.0, propulsor="fan"),
    "long_fan":                     Mission(range_nmi=6000.0, cruise_speed_kt=450.0, propulsor="fan"),
}

# ---------------------------------------------------------------------
# Quali missioni a quale sistema
#
#    Questa è la tabella da modificare per cambiare i punti
#    dell'analisi: aggiungere una voce qui aggiunge una colonna di Y e
#    quindi un grafico a nube. Ogni scelta è stata verificata sulla
#    frazione di NaN su un campione della PDF, non sul solo valore
#    nominale.
# ---------------------------------------------------------------------
REPRESENTATIVE_MISSIONS = {
    # dominio minuscolo: tre punti tutti dentro l'autonomia, uno per
    # isolare l'effetto della velocita' e uno per quello del range
    "Battery-electric": [
        "battery_very_short_low_speed",
        "battery_very_short",
        "battery_short",
        "battery_very_short_fan",
    ],
    # a elica copre tutto il piano; il punto con fan serve a vedere se
    # il ranking cambia quando cambia il propulsore, ed è corto apposta
    # (con fan a lungo raggio il fuel cell non converge in un quinto dei
    # campioni, e la colonna diventa piu' un test di fattibilita' che di
    # sensibilita')
    "Hydrogen fuel cell": [
        "short_low_speed_propeller",
        "short_high_speed_propeller",
        "medium_propeller",
        "long_propeller",
        "short_low_speed_fan",
        "short_high_speed_fan",
    ],
    # il sistema che domina la mappa: tassonomia completa con fan, più
    # un punto a elica
    "Hydrogen combustion": [
        "short_low_speed_fan",
        "short_high_speed_fan",
        "medium_fan",
        "long_fan",
        "medium_propeller",
    ],
    "e-SAF combustion": [
        "short_low_speed_fan",
        "medium_fan",
        "long_fan",
        "medium_propeller",
    ],
}


def _intensity(system_name, mission, systems, tech, sizer) -> float:
    """Intensity di un sistema, NaN se non fattibile o se il modello
    solleva (un theta patologico non deve far cadere il run)"""
    system = next(s for s in systems if s.name == system_name)
    try:
        return float(compute_intensity(mission, system, tech, sizer).intensity_MJ_per_pax_nmi)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class IntensityAt:
    """Renewable electricity intensity [MJ/(pax*nmi)] di un sistema in
    una missione.

    mission_label è l'etichetta della missione in MISSION_LIBRARY,
    tenuta per poter raggruppare e titolare i grafici: non entra nel
    calcolo.
    """
    system_name: str
    range_nmi: float
    speed_kt: float
    propulsor: str
    mission_label: str = ""

    @property
    def name(self) -> str:
        return (f"E[{self.system_name}]@{self.range_nmi:g}nmi_"
                f"{self.speed_kt:g}kt_{self.propulsor}")

    @property
    def mission(self) -> Mission:
        return Mission(range_nmi=self.range_nmi, cruise_speed_kt=self.speed_kt,
                       propulsor=self.propulsor)

    def evaluate(self, tech, wtt, systems, sizer) -> float:
        return _intensity(self.system_name, self.mission, systems, tech, sizer)


def outputs_from_missions(assignment: Optional[dict] = None,
                          library: Optional[dict] = None) -> list:
    """Costruisce la lista di IntensityAt da una tabella
    {nome_sistema: [etichette di missione]}.

    Serve a definire un insieme di punti diverso da quello di default
    senza riscrivere gli IntensityAt a mano:

        outputs = outputs_from_missions({"Hydrogen combustion":
                                         ["medium_fan", "long_fan"]})
    """
    assignment = assignment or REPRESENTATIVE_MISSIONS
    library = library or MISSION_LIBRARY

    outputs = []
    for system_name, labels in assignment.items():
        for label in labels:
            if label not in library:
                raise ValueError(f"Missione sconosciuta: {label!r}. Le missioni "
                                 f"disponibili sono {list(library)}")
            m = library[label]
            outputs.append(IntensityAt(system_name=system_name, range_nmi=m.range_nmi,
                                       speed_kt=m.cruise_speed_kt, propulsor=m.propulsor,
                                       mission_label=label))
    return outputs


def default_outputs() -> list:
    """L'insieme di partenza: la intensity di ciascun sistema nelle sue
    missioni rappresentative (REPRESENTATIVE_MISSIONS)"""
    return outputs_from_missions()


def outputs_table(outputs: Optional[list] = None) -> pd.DataFrame:
    """I punti scelti, in tabella. Da stampare in tesi accanto alla
    tabella delle condizioni operative: si vede a colpo
    d'occhio quali punti sono in comune fra analisi locale e globale, ed
    è su quelli che il confronto fra elasticità e mu* ha senso"""
    outputs = outputs or default_outputs()
    return pd.DataFrame([{"output": o.name, "sistema": o.system_name,
                          "missione": o.mission_label, "range_nmi": o.range_nmi,
                          "velocita_kt": o.speed_kt, "propulsore": o.propulsor}
                         for o in outputs])


def evaluate_outputs(theta: pd.DataFrame, outputs: Optional[list] = None,
                     verbose: bool = False, progress_every: int = 200) -> pd.DataFrame:
    """Valuta tutti gli output su tutte le righe di theta.

    Ritorna un DataFrame (n_campioni x n_output) con i nomi degli output
    come colonne. tech, wtt, sizer e la lista dei sistemi si costruiscono
    una volta per riga e si riusano per tutti gli output: è la ragione
    per cui conviene valutarli insieme invece che uno alla volta
    """
    outputs = outputs or default_outputs()
    names = [o.name for o in outputs]
    values = np.full((len(theta), len(outputs)), np.nan)

    for i in range(len(theta)):
        tech, wtt = theta_row_to_tech_wtt(theta.iloc[i].to_dict())
        sizer = AircraftSizer(tech)
        systems = build_default_systems(build_energy_carriers(wtt))
        for j, out in enumerate(outputs):
            try:
                values[i, j] = out.evaluate(tech, wtt, systems, sizer)
            except Exception:
                values[i, j] = np.nan
        if verbose and (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(theta)} valutazioni")

    return pd.DataFrame(values, columns=names)


def nan_report(Y: pd.DataFrame) -> pd.DataFrame:
    """Frazione di NaN per output. Da guardare prima degli indici: un
    output con molti NaN non è analizzabile, e il suo punto operativo va
    spostato dentro il dominio di fattibilità del sistema"""
    return (Y.isna().mean().rename("frazione_nan").to_frame()
            .assign(n_validi=Y.notna().sum())
            .sort_values("frazione_nan", ascending=False))
