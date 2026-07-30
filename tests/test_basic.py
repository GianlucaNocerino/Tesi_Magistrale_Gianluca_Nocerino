"""
Test di base del modello. Non sono test di validazione quantitativa
rispetto all'articolo (richiederebbero i dati numerici esatti delle
figure, che non sono disponibili), ma controlli di sanita' fisica:
il modello deve produrre numeri nell'ordine di grandezza giusto e con il
comportamento qualitativo descritto nell'articolo.

Esegui con: pytest tests/  (dalla cartella principale del progetto)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                   build_default_systems, build_energy_carriers, compute_intensity)


def _make_systems():
    tech = TechAssumptions()
    wtt = WellToTankEfficiencies()
    systems = build_default_systems(build_energy_carriers(wtt))
    return tech, systems


def test_short_range_battery_converges():
    """A corto raggio e bassa velocita' la batteria deve almeno convergere
    a un aeromobile fisicamente valido (Section 8, conclusioni).

    NOTA: questo test NON verifica che la batteria sia il sistema piu'
    efficiente in questo scenario, a differenza di quanto mostrato in
    Fig. 3/5 dell'articolo per corto raggio. Con le curve Fig. 9
    approssimate attualmente in propulsive_efficiency.py (vedi le
    avvertenze in quel file) il modello a volte fa risultare la fuel cell
    piu' efficiente della batteria anche a distanze molto brevi: e' il
    principale punto aperto da validare/tarare prima di usare il modello
    per conclusioni quantitative (si veda il README, sezione "Stato di
    validazione")."""
    tech, systems = _make_systems()
    mission = Mission(range_nmi=30, cruise_speed_kt=200, propulsor="propeller")

    results = {s.name: compute_intensity(mission, s, tech) for s in systems}
    battery = results["Battery-electric"]

    assert battery.sizing.converged
    assert not math.isnan(battery.intensity_MJ_per_pax_nmi)
    assert battery.intensity_MJ_per_pax_nmi > 0


def test_hydrogen_combustion_beats_esaf_everywhere():
    """L'idrogeno ha una produzione well-to-tank piu' efficiente dell'e-SAF
    a parita' di energia in volo, quindi la combustione a idrogeno deve
    richiedere meno elettricita' rinnovabile dell'e-SAF in ogni missione
    testata (Section 3 dell'articolo)."""
    tech, systems = _make_systems()
    by_name = {s.name: s for s in systems}

    for range_nmi in (200, 1000, 4000):
        for speed_kt, propulsor in ((250, "propeller"), (450, "fan")):
            mission = Mission(range_nmi=range_nmi, cruise_speed_kt=speed_kt, propulsor=propulsor)
            h2 = compute_intensity(mission, by_name["Hydrogen combustion"], tech)
            esaf = compute_intensity(mission, by_name["e-SAF combustion"], tech)
            assert h2.intensity_MJ_per_pax_nmi < esaf.intensity_MJ_per_pax_nmi


def test_empty_weight_fraction_is_physically_reasonable():
    """Per aeromobili convenzionali (e-SAF, nessuna correzione) la frazione
    di peso a vuoto deve restare in un intervallo plausibile (circa 0.4-0.7,
    coerente con velivoli commerciali reali, si veda Fig. 7)."""
    from cnav import T_to_W_definitions as TWdef

    for mtow in (10_000, 80_000, 250_000):
        frac = TWdef.oew_fraction_fan(mtow, TechAssumptions())
        assert 0.35 < frac < 0.75


def test_battery_electric_infeasible_at_long_range_is_flagged():
    """Su una missione molto lunga la batteria non deve dare un risultato
    fisicamente valido: il sizer deve segnalarlo come non convergente,
    non restituire un numero enorme senza avviso."""
    tech, systems = _make_systems()
    by_name = {s.name: s for s in systems}
    mission = Mission(range_nmi=5000, cruise_speed_kt=450, propulsor="fan")

    result = compute_intensity(mission, by_name["Battery-electric"], tech)
    assert not result.sizing.converged
    assert math.isnan(result.intensity_MJ_per_pax_nmi)


if __name__ == "__main__":
    test_short_range_battery_converges()
    test_hydrogen_combustion_beats_esaf_everywhere()
    test_empty_weight_fraction_is_physically_reasonable()
    test_battery_electric_infeasible_at_long_range_is_flagged()
    print("Tutti i test sono passati.")
