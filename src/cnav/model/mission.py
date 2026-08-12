from dataclasses import dataclass

from .units import NMI_TO_M

"""Definizione di una missione di volo."""


@dataclass(frozen=True)
class Mission:
    """Una missione è definita da range, velocità di crociera e dal tipo
    di propulsore (che determina la quota di crociera, le stime di
    peso e le efficienze usate nel modello).

    Attributes
    ----------
    range_nmi : float
        Range della missione [miglia nautiche].
    cruise_speed_kt : float
        Velocita' di crociera [nodi].
    propulsor : str
        "fan" (getto/ducted fan) oppure "propeller" (elica).
    """

    range_nmi: float
    cruise_speed_kt: float
    propulsor: str

    def __post_init__(self):
        if self.propulsor not in ("fan", "propeller"):
            raise ValueError("propulsor deve essere 'fan' o 'propeller'")

    @property
    def range_m(self) -> float:
        return self.range_nmi * NMI_TO_M

    @property
    def range_km(self) -> float:
        return self.range_m / 1000.0

    @property
    def cruise_speed_ms(self) -> float:
        from .units import KT_TO_MS
        return self.cruise_speed_kt * KT_TO_MS
