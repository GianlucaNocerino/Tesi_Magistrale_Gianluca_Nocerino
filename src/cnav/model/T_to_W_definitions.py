import math

from . import atmosphere
from . import propulsive_efficiency as pe
from .constants import TechAssumptions
from .mission import Mission
from .units import FT_TO_M, G, KT_TO_MS

LB_TO_KG = 0.45359237

"""
Di seguito tutte le definizioni necessarie per l'analisi Tank-to-Wake, divise in due cateegorie:
1) Funzioni fisiche condivise da tutti i sistemi propulsivi
2) Stime empiriche del peso, valide per velivoli convenzionali a jet fuel.
Sono il punto di partenza a cui ogni sistema propulsivo applica le proprie "correzioni"
"""

"""
Convenzione di unità di misura usata in questo modulo, per evitare errori:
- masse in kg
- potenze in kW (coerente con Table 1: kW/kg, N/kW, kg/kW)
- lunghezze/quote in m, velocita' in m/s
- energie in J (salvo dove specificato MJ)
"""

"""
1)
"""

def cruise_altitude_m(propulsor: str) -> float:
    """Quota di crociera: 25 000 ft per propeller, 35 000 ft per fan"""
    return (25_000.0 if propulsor == "propeller" else 35_000.0) * FT_TO_M


def _actual_altitude_m(mission: Mission, tech: TechAssumptions) -> float:
    """Quota effettivamente raggiunta durante la missione:
    può essere inferiore alla quota teorica di crociera se il range è
    troppo corto per completare la salita entro la prima metà del volo"""
    cruise_alt_m = cruise_altitude_m(mission.propulsor)
    climb_rate_ms = (tech.climb_rate_ft_per_min * FT_TO_M) / 60.0
    climb_speed_ms = tech.climb_speed_fraction * mission.cruise_speed_ms  # velocità di avanzamento in salita
    half_mission_time_s = 0.5 * mission.range_m / climb_speed_ms
    return min(cruise_alt_m, climb_rate_ms * half_mission_time_s)


def cruise_mach(mission: Mission, tech: TechAssumptions) -> float:
    return mission.cruise_speed_ms / atmosphere.speed_of_sound_ms(_actual_altitude_m(mission, tech))


def propulsive_efficiency(mission: Mission, tech: TechAssumptions) -> float:
    """Efficienza propulsiva effettiva alla velocità di crociera data:
    valore "fissato" (da constants, in tech) moltiplicato per il fattore di scala
    dipendente dal Mach"""
    mach = cruise_mach(mission, tech)
    if mission.propulsor == "fan":
        return tech.eta_p_fan * pe.fan_efficiency_scaling(mach, tech.fan_pressure_ratio, tech.fan_scaling_mach_ref)
    return tech.eta_p_propeller * pe.propeller_efficiency_scaling(
            mach, tech.propeller_curve_peak_mach, tech.propeller_curve_rise_rate,
            tech.propeller_curve_decay_width,
        )


def power_required_kW(mtow_kg: float, cruise_speed_ms: float, ld: float, propulsor: str) -> float:
    """Potenza installata richiesta, stimata proporzionale a MTOW*V/(L/D), con un fattore 
    moltiplicativo che dipende se il motore è con propeller o fan"""
    k_W_per_kg_per_ms = 40.0 if propulsor == "fan" else 20.0
    power_W = k_W_per_kg_per_ms * mtow_kg * cruise_speed_ms / ld
    return power_W / 1000.0


def drag_power_kW(mtow_kg: float, ld: float, cruise_speed_ms: float) -> float:
    """Potenza necessaria a vincere la resistenza in crociera (L=W in
    crociera, quindi D = W/(L/D))"""
    return (mtow_kg * G * cruise_speed_ms / ld) / 1000.0


def heat_rejected_kW(mtow_kg: float, ld: float, mission: Mission, tech: TechAssumptions,
                      loss_fraction: float) -> float:
    """Calore che il Thermal Management System (TMS) deve
    smaltire, a partire dalla potenza elettrica richiesta e dalla frazione
    di quella potenza persa come calore (loss_fraction, diversa per
    batteria e fuel cell)"""
    eta_p = propulsive_efficiency(mission, tech)
    electrical_power_kW = drag_power_kW(mtow_kg, ld, mission.cruise_speed_ms) / (eta_p * tech.eta_motor)
    return electrical_power_kW * loss_fraction


def tms_weight_kg(heat_kW: float, tech: TechAssumptions) -> float:
    """Peso del TMS, proporzionale al calore da smaltire"""
    return tech.kappa_TMS_kg_per_kW * heat_kW


def tms_drag_N(heat_kW: float, tech: TechAssumptions) -> float:
    """Resistenza aggiuntiva dovuta al TMS"""
    return tech.delta_TMS_N_per_kW * heat_kW


def ld_altitude_factor(mission: Mission, tech: TechAssumptions) -> float:
    """Fattore moltiplicativo su L/D per la quota effettivamente raggiunta.
 
    Ramo "fixed" (default): 1.0 sempre, cioè L/D nominale a prescindere
    dalla missione. Ramo "polar_decay": il decadimento vero.
 
    Con polare parabolica l'efficienza in funzione del C_L è
 
        E/E_max = 2 / (C_L/C_L_opt + C_L_opt/C_L)
 
    In crociera la portanza equilibra il peso, quindi a peso, superficie
    e velocità dati il C_L è inversamente proporzionale alla densità:
    volando più in basso del previsto si vola in aria più densa e il C_L
    di crociera scende sotto quello di massima efficienza. Posto
 
        xi = C_L/C_L_opt = rho_opt/rho_real <= 1
 
    si ottiene la forma chiusa dell'enunciato:
 
        E/E_max = 2*xi / (1 + xi^2)
 
    Il fattore è sempre <= 1 e vale 1 solo quando la quota di progetto
    viene raggiunta, cioè per tutte le missioni abbastanza lunghe. È
    quindi una penalità che si accende solo ai raggi corti, ed è
    esattamente la regione in cui vive la batteria: ci si aspetta che
    questa variante colpisca soprattutto lei.

    Semplificazione da dichiarare: si assume che il velivolo sia
    dimensionato per volare a E_max alla quota di progetto, cioè che
    C_L_opt corrisponda proprio alla condizione nominale"""

    if tech.model_form.aero_model != "polar_decay":
        return 1.0
 
    rho_opt = atmosphere.density_kg_per_m3(cruise_altitude_m(mission.propulsor))
    rho_real = atmosphere.density_kg_per_m3(_actual_altitude_m(mission, tech))
    if rho_real <= 0.0:
        return 1.0
 
    xi = rho_opt / rho_real
    return 2.0 * xi / (1.0 + xi ** 2)


def ld_with_extra_drag(ld_baseline: float, mtow_kg: float, extra_drag_N: float) -> float:
    """L/D risultante aggiungendo una resistenza extra costante
    (ad esempio nel caso del contributo aggiuntivo dovuto al TMS)"""
    ld = 1.0 / (1.0 / ld_baseline + extra_drag_N / (mtow_kg * G))
    return max(ld, 1.0)


def hydrogen_tank_weight_kg(fuel_weight_kg: float, gamma_tank: float) -> float:
    """Peso del serbatoio di idrogeno liquido, proporzionale al
    peso dell'idrogeno liquido trasportato (inclusa la riserva)"""
    return fuel_weight_kg * (1.0 - gamma_tank) / gamma_tank


def _reserve_range_paper_m(mission: Mission, tech: TechAssumptions) -> float:
    """Riserva del modello base: rotta verso l'alternativo + loiter.
 
    La velocità di loiter usa tech.reserve_loiter_speed_kt se specificata
    (valore numerico); di default (NaN) usa la velocità di crociera della
    missione, come assunto nell'articolo di riferimento.
 
    È l'ipotesi accademica: attendere alla velocità di crociera è la cosa
    più dispendiosa che si possa fare, e nessun operatore lo fa. Il ramo
    "easa" corregge proprio questo"""

    alternate_m = min(mission.range_m, tech.reserve_alternate_range_nmi * 1852.0)
    if math.isnan(tech.reserve_loiter_speed_kt):
        loiter_speed_ms = mission.cruise_speed_ms
    else:
        loiter_speed_ms = tech.reserve_loiter_speed_kt * KT_TO_MS
    loiter_m = tech.reserve_loiter_time_s * loiter_speed_ms
    return alternate_m + loiter_m
 
 
def _reserve_range_easa_m(mission: Mission, tech: TechAssumptions) -> float:
    """Riserva secondo i vincoli operativi reali (EASA CAT.OP.MPA.150).
 
    Tre differenze rispetto al ramo base:
 
      contingenza   il 5% del combustibile di tratta. Qui è applicato
                    come 5% di rotta in più, che per Breguet è la stessa
                    cosa a meno di termini del secondo ordine, e ha il
                    vantaggio di restare dentro la sola grandezza che il
                    sizer maneggia (una distanza)
      riserva finale  30 minuti per i fan, 45 per gli elica. Il modello
                    base ne usa 45 per tutti
      velocità      la riserva finale si vola alla velocità di massima
                    autonomia oraria, presa come frazione di quella di
                    crociera, non alla velocità di crociera
 
    L'alternativo resta quello del modello base (200 nmi), perché è già
    una regola operativa e non un'assunzione accademica.
 
    ATTENZIONE al segno dell'effetto: il termine di contingenza fa
    crescere la riserva, il loiter più lento e (per i fan) più corto la
    fa calare. Quale dei due vince dipende dal range, quindi questa
    variante non sposta la mappa tutta nella stessa direzione: sui
    raggi corti domina il loiter (la riserva è quasi tutto), sui lunghi
    domina la contingenza (il 5% di una rotta lunga). È il motivo per
    cui vale la pena guardarla sulla mappa e non su un punto solo"""

    contingency_m = tech.reserve_contingency_fraction * mission.range_m
    alternate_m = min(mission.range_m, tech.reserve_alternate_range_nmi * 1852.0)
 
    final_time_s = (tech.reserve_final_time_fan_s if mission.propulsor == "fan"
                    else tech.reserve_final_time_propeller_s)
    loiter_speed_ms = tech.reserve_loiter_speed_fraction * mission.cruise_speed_ms
    final_m = final_time_s * loiter_speed_ms
 
    return contingency_m + alternate_m + final_m
 
 
def reserve_range_m(mission: Mission, tech: TechAssumptions) -> float:
    """Range equivalente della missione di riserva.
 
    Bivio su tech.model_form.reserve_model: "paper" (default) oppure
    "easa". Vedi model_form.py"""
    if tech.model_form.reserve_model == "easa":
        return _reserve_range_easa_m(mission, tech)
    return _reserve_range_paper_m(mission, tech)


def breguet_fuel_weight_kg(mtow_kg: float, ld: float, eta_overall: float,
                            specific_energy_J_per_kg: float, distance_m: float) -> float:
    """Equazione di Breguet classica, dati il range (distance_m), l'efficienza globale
    del sistema propulsivo (eta_overall) e l'efficienza aerodinamica (ld)"""
    import math
    exponent = distance_m * G / (eta_overall * specific_energy_J_per_kg * ld)
    return mtow_kg * (1.0 - math.exp(-exponent))


def climb_energy_weight_kg(mtow_kg: float, mission: Mission, eta_overall: float,
                            specific_energy_J_per_kg: float, tech: TechAssumptions) -> float:
    """Peso dell'energia (combustibile o batteria) necessaria
    per la salita, stimata come la somma degli incrementi di energia cinetica 
    e potenziale, convertita in massa attraverso l'efficienza propulsiva globale.

    Nota / semplificazione: l'articolo descrive questo calcolo a parole ma
    non riporta un'equazione esplicita (a differenza delle altre
    sezioni). Qui la resistenza aerodinamica durante la salita è trascurata
    (è un'assunzione comune nei modelli concettuali di questo tipo, ma
    andrebbe verificata se serve una riproduzione quantitativa esatta).
    """
    climb_alt_m = _actual_altitude_m(mission, tech)

    delta_pe_J_per_kg = G * climb_alt_m
    delta_ke_J_per_kg = 0.5 * mission.cruise_speed_ms ** 2
    mechanical_energy_J = mtow_kg * (delta_pe_J_per_kg + delta_ke_J_per_kg)

    return mechanical_energy_J / (eta_overall * specific_energy_J_per_kg)


def turbofan_overall_efficiency(power_per_engine_MW: float) -> float:
    """Efficienza complessiva (termica x propulsiva) di un turbofan,
    in funzione della potenza per singolo motore (potenza totale divisa per numero di motori)"""
    import math
    return 1e-2 * (9.840 * math.log10(power_per_engine_MW) + 20.407)


def turboprop_thermal_efficiency(power_per_engine_MW: float) -> float:
    """Efficienza termica del nucleo di un turboelica, in funzione
    della potenza per singolo motore."""
    import math
    return 1e-2 * (9.371 * math.log10(power_per_engine_MW) + 28.813)


def combustion_overall_efficiency(mission: Mission, tech: TechAssumptions,
                                   power_req_kW: float) -> float:
    """Efficienza complessiva di un motore a combustione (idrogeno o e-SAF, fan o propeller).
    Si assume una configurazione bimotore"""
    power_per_engine_MW = max((power_req_kW / 2.0) / 1000.0, 1e-6)
    mach = cruise_mach(mission, tech)
    if mission.propulsor == "fan":
        eta_base = turbofan_overall_efficiency(power_per_engine_MW)
        scale = pe.fan_efficiency_scaling(mach, tech.fan_pressure_ratio, tech.fan_scaling_mach_ref)
    else:
        eta_thermal = turboprop_thermal_efficiency(power_per_engine_MW)
        eta_base = eta_thermal * tech.eta_p_propeller
        scale = pe.propeller_efficiency_scaling(
            mach, tech.propeller_curve_peak_mach, tech.propeller_curve_rise_rate,
            tech.propeller_curve_decay_width,
        )
    return eta_base * scale


"""
2)
"""

def oew_fraction_fan(mtow_kg: float, tech: TechAssumptions) -> float:
    """Stima empirica del OEW/MTOW per velivoli a getto/ducted fan."""
    return tech.oew_fan_a * mtow_kg ** tech.oew_fan_b + tech.oew_fan_c


def _oew_fraction_propeller_raw(mtow_kg: float, tech: TechAssumptions) -> float:
    """Stima empirica 'grezza' per velivoli ad elica.

    Nota: pochi velivoli ad elica commerciali sono pesanti quanto i
    grandi jet, quindi questa stima, da sola, dnon è valida per MTOW elevati. 
    Per questo esiste oew_fraction_propeller, che la estende in modo continuo"""
    return tech.oew_prop_a * mtow_kg ** tech.oew_prop_b + tech.oew_prop_c


def oew_fraction_propeller(mtow_kg: float, tech: TechAssumptions) -> float:
    """Combinazione tra la stima per l'elica e quella del fan,
    per estendere in modo fisicamente ragionevole la stima empirica per l'elica
    a MTOW elevati"""
    f_fan = oew_fraction_fan(mtow_kg, tech)
    f_prop = _oew_fraction_propeller_raw(mtow_kg, tech)
    return (1.0 / 40.0) * math.log(math.exp(40 * f_fan) + math.exp(40 * f_prop))

 
def oew_fraction_raymer(mtow_kg: float, propulsor: str, tech: TechAssumptions) -> float:
    """Frazione di peso a vuoto secondo Raymer: We/W0 = A * W0^C.
 
    ATTENZIONE ALLE UNITÀ: la correlazione di Raymer è tarata con W0 in
    libbre, e l'esponente non è adimensionale, quindi darle dei kg
    produce un numero sbagliato ma perfettamente plausibile (con
    C = -0.06 l'errore è un fattore 2.2^-0.06, circa il 5%: abbastanza
    piccolo da non saltare all'occhio e abbastanza grande da sporcare i
    confronti). La conversione avviene qui dentro, una volta sola.
 
    Differenze concettuali rispetto al ramo "paper", che sono poi la
    ragione per cui questa variante è interessante:
 
      forma      potenza pura, senza il termine costante c. La curva del
                 paper ha un asintoto (tende a c per MTOW grandi),
                 questa no: decresce indefinitamente, molto lentamente
      fonte      coefficienti da tabella di manuale per classe di
                 velivolo, non da una regressione fatta sui dati. Sono
                 quindi tarati su velivoli convenzionali a jet fuel,
                 esattamente come quelli del paper, ma da un campione
                 diverso e con un fit diverso
      parametri  spariscono b, c e r_pivot per entrambi i propulsori.
                 Sei parametri incerti in meno, due nuovi (A_fan,
                 A_prop). È il caso che rende necessario dividere i
                 parametri in comuni e specifici del modello: vedi
                 model_form_uncertainty.py
 
    Non c'è raccordo fra i due propulsori come in oew_fraction_propeller:
    le classi di Raymer sono separate per costruzione e la correlazione
    dell'elica non ha il problema di estrapolazione ad alto MTOW che ha
    quella del paper (l'esponente è negativo, la funzione resta
    positiva e monotona ovunque)"""
    mtow_lb = mtow_kg / LB_TO_KG
    if propulsor == "fan":
        return tech.raymer_A_fan * mtow_lb ** tech.raymer_C_fan
    return tech.raymer_A_prop * mtow_lb ** tech.raymer_C_prop
 
 
def oew_fraction(mtow_kg: float, propulsor: str, tech: TechAssumptions) -> float:
    """Sceglie la stima giusta in base al tipo di propulsore e al ramo
    attivo in tech.model_form.oew_model ("paper" di default, "raymer")"""
    if propulsor not in ("fan", "propeller"):
        raise ValueError(f"propulsore sconosciuto: {propulsor!r} (atteso 'fan' o 'propeller')")
 
    if tech.model_form.oew_model == "raymer":
        return oew_fraction_raymer(mtow_kg, propulsor, tech)
 
    if propulsor == "fan":
        return oew_fraction_fan(mtow_kg, tech)
    return oew_fraction_propeller(mtow_kg, tech)


def payload_weight_kg(range_km: float, tech: TechAssumptions) -> float:
    """Peso del payload stimato in funzione del range di missione (no riserva), 
    sulla base di una regressione realizzata sui dati dei velivoli commerciali esistenti"""
    return (tech.payload_a * range_km ** tech.payload_b + tech.payload_c) \
        * (tech.payload_d * range_km + tech.payload_e)