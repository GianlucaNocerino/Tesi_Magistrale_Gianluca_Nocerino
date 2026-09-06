"""Modello atmosferico: velocità del suono e densità"""


def speed_of_sound_ms(altitude_m: float) -> float:
    """Velocità del suono approssimata con un fit lineare della ISA fino
    alla tropopausa (11 km), poi costante"""
    return max(340.3 + (295.1 - 340.3) / 11000.0 * altitude_m, 295.1)

def density_kg_per_m3(altitude_m: float) -> float:
    """Densità ISA, troposfera (gradiente -6.5 K/km) fino a 11 km e
    stratosfera isoterma oltre.
 
    Serve solo al ramo aero_model="polar_decay", che confronta la
    densità alla quota effettivamente raggiunta con quella alla quota
    di progetto.
    """
    rho0 = 1.225
    if altitude_m <= 11_000.0:
        return rho0 * (1.0 - 2.25577e-5 * altitude_m) ** 4.2559
    rho11 = rho0 * (1.0 - 2.25577e-5 * 11_000.0) ** 4.2559
    return rho11 * math.exp(-(altitude_m - 11_000.0) / 6341.6)