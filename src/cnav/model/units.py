"""
Conversioni di unità di misura usate in tutto il modello.

Teniamo tutte le grandezze fisiche "interne" al modello in unità SI
(kg, m, s, W, J) e convertiamo solo in ingresso/uscita, così da
evitare errori di coerenza dimensionale nelle formule.
"""

NMI_TO_M = 1852.0          # 1 miglio nautico in metri
KT_TO_MS = 0.514444        # 1 nodo (knot) in m/s
FT_TO_M = 0.3048           # 1 piede in metri
WH_TO_J = 3600.0           # 1 Wh in Joule
G = 9.80665                # accelerazione di gravita' standard [m/s^2]
