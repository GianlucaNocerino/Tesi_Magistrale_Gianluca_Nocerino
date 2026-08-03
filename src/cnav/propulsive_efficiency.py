"""
Scaling Factor dell'efficienza propulsiva in funzione del Mach di volo.

FAN - formula reale trovata nel riferimento citato dall'articolo:
    Michel, U. (2011), "The benefits of variable area fan nozzles on
    turbofan engines", AIAA 2011-226 [rif. 55 di Adler & Martins].

Michel fornisce l'efficienza propulsiva di un fan senza perdite
in funzione del Mach di volo M_f e del Mach del getto M_j,
dove M_j si ricava dal Mach di volo e dal rapporto di compressione del fan,
che non e' un parametro elencato nella Table 1 di Adler & Martins 
(assunto da me di default 1.5, lo stesso valore usato
come esempio da Michel in tutto il suo articolo, ma modificabile in TechAssumptions)

ELICA - nessuna formula disponibile nel riferimento citato:
    Alves, P., Silvestre, M., Gamboa, P. (2020), "Aircraft Propellers -
    Is There a Future?", Energies 13(16), 4157 [rif. 56 di Adler & Martins].

La funzione di seguito resta perciò un'approssimazione, ottenuta valutando
l'andamento dello scaling factor per il rpopeller mostrato da Adler & Martins.
"""
import math


def fan_efficiency_scaling(mach: float, fan_pressure_ratio: float = 1.5, gamma: float = 1.4) -> float:
    """Efficienza propulsiva di un fan senza perdite, in funzione del Mach 
    di volo e del rapporto di compressione del fan"""
    if mach <= 0.0:
        return 0.0
    exponent = (gamma - 1.0) / gamma
    """Efficienza propulsiva al Mach corrente"""
    jet_mach_sq = ((fan_pressure_ratio ** exponent) * (1.0 + (gamma - 1.0) / 2.0 * mach ** 2) - 1.0) \
        * 2.0 / (gamma - 1.0)
    jet_mach = math.sqrt(max(jet_mach_sq, 0.0))
    eta_p_current = 2.0 * mach / (jet_mach + mach)
    """Efficienza propulsiva al Mach di riferimento (0.8)"""
    mach_ref = 0.8
    jet_mach_sq_ref = ((fan_pressure_ratio ** exponent) * (1.0 + (gamma - 1.0) / 2.0 * mach_ref ** 2) - 1.0) \
        * 2.0 / (gamma - 1.0)
    jet_mach_ref = math.sqrt(max(jet_mach_sq_ref, 0.0))
    eta_p_ref = 2.0 * mach_ref / (jet_mach_ref + mach_ref)
    """Lo scaling Factor è l'efficienza propulsiva (Michel, [55]) normalizzata per il suo valore
    a Mach=0.8"""
    return eta_p_current / eta_p_ref


def propeller_efficiency_scaling(mach: float, peak_mach: float = 0.4, rise_rate: float = 14.0,
                                  decay_width: float = 0.10) -> float:
    """Approssimazione, sale come 1-exp(-rise_rate*mach) fino al picco in
    peak_mach, poi cala come una gaussiana di larghezza decay_width.
    I tre parametri di forma sono letti da TechAssumptions
    così da poterli far variare"""
    peak_value = 1.0 - math.exp(-rise_rate * peak_mach)
    if mach <= peak_mach:
        return 1.0 - math.exp(-rise_rate * mach)
    return peak_value * math.exp(-((mach - peak_mach) / decay_width) ** 2)
