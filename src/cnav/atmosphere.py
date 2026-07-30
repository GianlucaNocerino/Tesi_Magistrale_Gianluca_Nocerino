"""Modello atmosferico: velocità del suono"""


def speed_of_sound_ms(altitude_m: float) -> float:
    """Velocità del suono approssimata con un fit lineare della ISA fino
    alla tropopausa (11 km), poi costante"""
    return max(340.3 + (295.1 - 340.3) / 11000.0 * altitude_m, 295.1)
