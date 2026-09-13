"""
Effetti dell'energy mix: dall'elettricità rinnovabile alla fonte alle
emissioni di CO2 e all'energia primaria.

Il modello deterministico si ferma all'elettricità richiesta alla
fonte, assumendola implicitamente rinnovabile e quindi a emissioni
nulle. Questo script aggiunge lo strato che sta ancora più a monte:
dato un energy mix, riassunto da un rendimento di catena e da
un'intensità di carbonio, entrambi già mediati su tutte le fonti,
(rinnovabili e non) calcola due nuovi output per passeggero-miglio nautico:

  1. energia primaria richiesta          [MJ/(pax*nmi)]
  2. anidride carbonica emessa           [g/(pax*nmi)]

La valutazione è fatta su una serie di missioni scelte in modo che il
sistema propulsivo carbon-neutral vincente non sia sempre lo stesso
(vedi MISSIONI più sotto): per ciascuna missione lo script cerca da
solo il sistema migliore con most_efficient_system, e poi confronta:

  - il migliore sistema carbon-neutral per quella missione;
  - l'e-SAF, che è il drop-in sostenibile sulla stessa cellula
    convenzionale (raramente è il migliore, ma è il termine di
    paragone piu' vicino al cherosene);
  - il velivolo convenzionale a cherosene, identico all'e-SAF nel
    tank-to-wake e diverso solo nella catena a monte.

Uso:
    python execution/11_aggiunte_well_to_tank/energy_mix_effects.py

Output (nella cartella di questo script):
    energy_mix_effects.csv     una riga per missione e per sistema
    energy_mix_co2.png         CO2 per pax*nmi, sostenibile vs cherosene
    energy_mix_primary.png     energia primaria per pax*nmi
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import (ESAFCombustion, EnergyCarrier, Mission, TechAssumptions,
                  WellToTankEfficiencies, build_default_systems,
                  build_energy_carriers, compute_intensity,
                  most_efficient_system)

OUT_DIR = Path(__file__).resolve().parent


# =====================================================================
# 1. INPUT
# =====================================================================

# ---------------------------------------------------------------------
# L'energy mix, riassunto in due soli numeri.
#
# Entrambi sono già valutati sull'insieme di tutte le fonti primarie.
#
#   ETA_CATENA        rendimento medio di catena dalla fonte primaria
#                     all'elettricità sulla rete: MJ elettrici prodotti 
#                     per MJ energia primaria impiegata. L'energia primaria
#                     richiesta è semplicemente l'elettricitaà divisa
#                     per questo rendimento.
#
#   GCO2_PER_MJ_EL    intensità di carbonio dell'elettricità: grammi
#                     di CO2 emessi per ogni MJ di elettricità
#                     prodotta. Se il dato di partenza è in gCO2/kWh, 
#                     dividerlo per 3.6.
#
# Conseguenza utile: entrambi gli output sono lineari nell'elettricità
# richiesta alla fonte, quindi raddoppiare l'intensità di carbonio
# raddoppia esattamente la CO2 di ogni missione, e la soglia di
# pareggio con il cherosene si ricava per proporzione.
#
# ---------------------------------------------------------------------
ETA_CATENA = 0.487
GCO2_PER_MJ_EL = 92.4      # 46.5 g/MJ = 167.4 g/kWh

# ---------------------------------------------------------------------
# Il riferimento convenzionale a cherosene (Jet A-1).
#
#   lhv_J_per_kg          potere calorifico inferiore
#   eta_well_to_tank      MJ di cherosene al serbatoio per MJ di
#                         petrolio greggio estratto (estrazione,
#                         raffinazione, trasporto)
#   gCO2_per_MJ_burn      CO2 della combustione a bordo, per MJ di
#                         cherosene bruciato (3150 g di CO2 per kg di
#                         carburante diviso 43 MJ/kg da circa 73)
#   gCO2_per_MJ_upstream  CO2 della filiera a monte, per MJ di
#                         cherosene al serbatoio
#
# Anche questi sono PLACEHOLDER da sostituire.
# ---------------------------------------------------------------------
KEROSENE_LHV_J_PER_KG = 43.0e6
KEROSENE_ETA_WELL_TO_TANK = 0.900455
KEROSENE_GCO2_PER_MJ_BURN = 73.2
KEROSENE_GCO2_PER_MJ_UPSTREAM = 7.402

# ---------------------------------------------------------------------
# Le missioni.
#
# Scelte sulla mappa best-system in modo da diversificare il vincitore:
# i primi punti stanno nel dominio della batteria, quelli centrali in
# quello della fuel cell, gli ultimi in quello della combustione a
# idrogeno, che è il sistema che occupa la fetta più grande del
# piano. Lo script non da per scontato chi vince, ma lo ricalcola e lo
# stampa, quindi se la calibrazione cambia i confini la tabella resta
# corretta (cambierà solo la diversificazione).
# ---------------------------------------------------------------------
MISSIONI = [
    ("Regionale cortissimo",  Mission(range_nmi=10.0,    cruise_speed_kt=200.0, propulsor="propeller")),
    ("Feeder elica",          Mission(range_nmi=50.0,    cruise_speed_kt=250.0, propulsor="propeller")),
    ("Regionale elica",       Mission(range_nmi=300.0,   cruise_speed_kt=250.0, propulsor="propeller")),
    ("Medio raggio elica",    Mission(range_nmi=1500.0,  cruise_speed_kt=250.0, propulsor="propeller")),
    ("Corto raggio fan",      Mission(range_nmi=100.0,   cruise_speed_kt=450.0, propulsor="fan")),
    ("Medio raggio fan",      Mission(range_nmi=1500.0,  cruise_speed_kt=450.0, propulsor="fan")),
    ("Lungo raggio fan",      Mission(range_nmi=6000.0,  cruise_speed_kt=450.0, propulsor="fan")),
    ("Ultra-lungo fan",       Mission(range_nmi=10000.0, cruise_speed_kt=450.0, propulsor="fan")),
]


# =====================================================================
# 2. LE DUE CATENE A MONTE
# =====================================================================

class EnergyMix:
    """L'energy mix visto dal lato dell'elettricità prodotta.

    Traduce i MJ elettrici richiesti alla fonte, che sono l'output del
    modello deterministico, nei due nuovi output: energia primaria e
    CO2. Entrambe le conversioni sono un solo prodotto, perchè il
    rendimento e l'intensità di carbonio sono già medie su tutte le
    fonti
    """

    def __init__(self, eta_catena: float, gCO2_per_MJ_el: float):
        if not 0.0 < eta_catena <= 1.0:
            raise ValueError(f"ETA_CATENA deve stare in (0, 1], non {eta_catena}")
        self.eta_catena = eta_catena
        self.gCO2_per_MJ_el = gCO2_per_MJ_el

    def primary_energy_MJ(self, electricity_MJ: float) -> float:
        return electricity_MJ / self.eta_catena

    def co2_g(self, electricity_MJ: float) -> float:
        return electricity_MJ * self.gCO2_per_MJ_el

    @property
    def carbon_intensity_g_per_kWh(self) -> float:
        """Lo stesso fattore di emissione nell'unità in cui di solito è
        tabulato"""
        return self.gCO2_per_MJ_el * 3.6


class FossilFuelChain:
    """La catena a monte del cherosene convenzionale: dal greggio al
    serbatoio, più la combustione a bordo.

    A differenza dell'EnergyMix non parte dall'elettricità ma
    dall'energia di combustibile effettivamente imbarcata.
    """

    def __init__(self, eta_well_to_tank: float, gCO2_per_MJ_burn: float,
                 gCO2_per_MJ_upstream: float):
        self.eta_well_to_tank = eta_well_to_tank
        self.gCO2_per_MJ_burn = gCO2_per_MJ_burn
        self.gCO2_per_MJ_upstream = gCO2_per_MJ_upstream

    def primary_energy_MJ(self, fuel_energy_MJ: float) -> float:
        return fuel_energy_MJ / self.eta_well_to_tank

    def co2_g(self, fuel_energy_MJ: float) -> float:
        return fuel_energy_MJ * (self.gCO2_per_MJ_burn + self.gCO2_per_MJ_upstream)


class KeroseneCombustion(ESAFCombustion):
    """Il velivolo convenzionale a cherosene.

    Nel tank-to-wake è identico all'e-SAF (stessa cellula, stesso
    motore, stesso OEW: l'e-SAF e' un drop-in), e differisce solo per il
    potere calorifico e, soprattutto, per cosa c'è a monte del
    serbatoio. Il carrier ha efficienza 1.0 perchè la catena a monte
    del cherosene non passa per l'elettricità e viene gestita a parte
    da FossilFuelChain.
    """

    name = "Kerosene combustion"

    def __init__(self, lhv_J_per_kg: float = KEROSENE_LHV_J_PER_KG):
        super().__init__(EnergyCarrier("Jet A-1 (fossile)", 1.0))
        self.lhv_J_per_kg = lhv_J_per_kg

    def specific_energy_J_per_kg(self, tech) -> float:
        return self.lhv_J_per_kg


# =====================================================================
# 3. VALUTAZIONE DI UNA MISSIONE
# =====================================================================

@dataclass
class RowResult:
    """Una riga della tabella: un sistema propulsivo su una missione"""

    mission_label: str
    range_nmi: float
    cruise_speed_kt: float
    propulsor: str
    system_name: str
    role: str                       # "best", "drop-in" oppure "convenzionale"
    energy_at_tank_MJ_per_pax_nmi: float
    electricity_MJ_per_pax_nmi: float   # NaN per il cherosene
    primary_MJ_per_pax_nmi: float
    co2_g_per_pax_nmi: float


def _per_pax_nmi(value: float, result, mission: Mission) -> float:
    """Normalizza una grandezza di missione per passeggero-miglio"""
    if not result.sizing.converged or result.n_pax <= 0:
        return float("nan")
    return value / (result.n_pax * mission.range_nmi)


def evaluate_sustainable(mission_label: str, mission: Mission, system, mix: EnergyMix,
                          tech: TechAssumptions, role: str) -> RowResult:
    """Un sistema carbon-neutral: si parte dall'elettricità alla fonte
    già calcolata dal modello e si applica l'energy mix"""
    result = compute_intensity(mission, system, tech)
    electricity_MJ = result.renewable_electricity_MJ

    return RowResult(
        mission_label=mission_label,
        range_nmi=mission.range_nmi,
        cruise_speed_kt=mission.cruise_speed_kt,
        propulsor=mission.propulsor,
        system_name=system.name,
        role=role,
        energy_at_tank_MJ_per_pax_nmi=_per_pax_nmi(result.energy_at_tank_MJ, result, mission),
        electricity_MJ_per_pax_nmi=result.intensity_MJ_per_pax_nmi,
        primary_MJ_per_pax_nmi=_per_pax_nmi(mix.primary_energy_MJ(electricity_MJ), result, mission),
        co2_g_per_pax_nmi=_per_pax_nmi(mix.co2_g(electricity_MJ), result, mission),
    )


def evaluate_kerosene(mission_label: str, mission: Mission, fossil: FossilFuelChain,
                       tech: TechAssumptions) -> RowResult:
    """Il velivolo convenzionale: si parte dall'energia di combustibile
    imbarcata e si applica la catena fossile"""
    system = KeroseneCombustion()
    result = compute_intensity(mission, system, tech)
    fuel_MJ = result.energy_at_tank_MJ

    return RowResult(
        mission_label=mission_label,
        range_nmi=mission.range_nmi,
        cruise_speed_kt=mission.cruise_speed_kt,
        propulsor=mission.propulsor,
        system_name=system.name,
        role="convenzionale",
        energy_at_tank_MJ_per_pax_nmi=_per_pax_nmi(fuel_MJ, result, mission),
        electricity_MJ_per_pax_nmi=float("nan"),
        primary_MJ_per_pax_nmi=_per_pax_nmi(fossil.primary_energy_MJ(fuel_MJ), result, mission),
        co2_g_per_pax_nmi=_per_pax_nmi(fossil.co2_g(fuel_MJ), result, mission),
    )


def build_table(missioni: list, mix: EnergyMix, fossil: FossilFuelChain,
                 tech: TechAssumptions, wtt: WellToTankEfficiencies) -> pd.DataFrame:
    """Per ogni missione: sistema migliore, e-SAF, cherosene"""
    rows = []
    for label, mission in missioni:
        systems = build_default_systems(build_energy_carriers(wtt))
        best_name = most_efficient_system(mission, systems, tech)["best"]
        if best_name is None:
            print(f"  [avviso] '{label}': nessun sistema converge, missione saltata")
            continue

        by_name = {s.name: s for s in systems}
        rows.append(evaluate_sustainable(label, mission, by_name[best_name], mix, tech, "best"))
        if best_name != "e-SAF combustion":
            rows.append(evaluate_sustainable(label, mission, by_name["e-SAF combustion"],
                                              mix, tech, "drop-in"))
        rows.append(evaluate_kerosene(label, mission, fossil, tech))

    return pd.DataFrame([r.__dict__ for r in rows])


# =====================================================================
# 4. STAMPA E GRAFICI
# =====================================================================

def print_mix_summary(mix: EnergyMix, fossil: FossilFuelChain) -> None:
    print("=" * 78)
    print("ENERGY MIX")
    print("=" * 78)
    print(f"Rendimento di catena            : {mix.eta_catena:.3f} MJ di elettricità per MJ alla fonte")
    print(f"Intensità di carbonio: {mix.gCO2_per_MJ_el:.2f} g/MJ "
          f"({mix.carbon_intensity_g_per_kWh:.1f} g/kWh)")
    print(f"\nCherosene: eta_wtt = {fossil.eta_well_to_tank:.2f}, "
          f"combustione = {fossil.gCO2_per_MJ_burn:.1f} g/MJ, "
          f"a monte = {fossil.gCO2_per_MJ_upstream:.1f} g/MJ")


def print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("RISULTATI PER MISSIONE  [per passeggero-miglio nautico]")
    print("=" * 78)
    for label, block in df.groupby("mission_label", sort=False):
        head = block.iloc[0]
        print(f"\n{label}  ({head.range_nmi:.0f} nmi, {head.cruise_speed_kt:.0f} kt, {head.propulsor})")
        print(f"  {'sistema':<26}{'ruolo':<15}{'primaria':>12}{'CO2':>12}")
        print(f"  {'':<26}{'':<15}{'[MJ]':>12}{'[g]':>12}")
        for _, row in block.iterrows():
            print(f"  {row.system_name:<26}{row.role:<15}"
                  f"{row.primary_MJ_per_pax_nmi:>12.3f}{row.co2_g_per_pax_nmi:>12.1f}")

        best = block[block.role == "best"].iloc[0]
        conv = block[block.role == "convenzionale"].iloc[0]
        if conv.co2_g_per_pax_nmi > 0:
            # variazione percentuale della CO2 del migliore sostenibile
            # rispetto al cherosene: negativa = emette meno
            variazione = 100.0 * (best.co2_g_per_pax_nmi / conv.co2_g_per_pax_nmi - 1.0)
            verso = "in meno" if variazione < 0 else "in più"
            print(f"  -> il migliore sostenibile emette il {abs(variazione):.1f}% {verso} "
                  f"del cherosene")


def plot_comparison(df: pd.DataFrame, column: str, ylabel: str, title: str,
                     filename: str) -> Path:
    """Barre affiancate: migliore sostenibile, e-SAF, cherosene"""
    labels = list(dict.fromkeys(df.mission_label))
    ruoli = ["best", "drop-in", "convenzionale"]
    colori = {"best": "#4C72B0", "drop-in": "#C44E52", "convenzionale": "#8C8C8C"}

    x = np.arange(len(labels))
    larghezza = 0.26

    fig, ax = plt.subplots(figsize=(11, 6))
    for k, ruolo in enumerate(ruoli):
        valori = []
        for label in labels:
            sel = df[(df.mission_label == label) & (df.role == ruolo)]
            valori.append(float(sel[column].iloc[0]) if len(sel) else np.nan)
        ax.bar(x + (k - 1) * larghezza, valori, larghezza,
               label=ruolo, color=colori[ruolo])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =====================================================================
# 5. MAIN
# =====================================================================

def main() -> None:
    tech = TechAssumptions()
    wtt = WellToTankEfficiencies()

    mix = EnergyMix(ETA_CATENA, GCO2_PER_MJ_EL)
    fossil = FossilFuelChain(KEROSENE_ETA_WELL_TO_TANK, KEROSENE_GCO2_PER_MJ_BURN,
                             KEROSENE_GCO2_PER_MJ_UPSTREAM)

    print_mix_summary(mix, fossil)
    df = build_table(MISSIONI, mix, fossil, tech, wtt)
    print_table(df)

    csv_path = OUT_DIR / "energy_mix_effects.csv"
    df.to_csv(csv_path, index=False)

    p1 = plot_comparison(df, "co2_g_per_pax_nmi", "CO2 [g/(pax*nmi)]",
                         "Emissioni di CO2: sostenibile vs convenzionale",
                         "energy_mix_co2.png")
    p2 = plot_comparison(df, "primary_MJ_per_pax_nmi", "Energia primaria [MJ/(pax*nmi)]",
                         "Energia primaria richiesta: sostenibile vs convenzionale",
                         "energy_mix_primary.png")

    print(f"\nTabella salvata in: {csv_path}")
    print(f"Figure salvate in : {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
