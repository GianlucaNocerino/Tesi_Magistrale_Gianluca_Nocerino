"""
Demo degli strumenti di sensibilità "continui" (superficie 3D
dell'elasticità sul piano range-velocità), quelli non già mostrati in local_sensitivity_report.py
(Fase 3, punti operativi discreti) né derivabili da un singolo tornado
plot. run_linear_sensitivity resta utile per un controllo rapido
ad-hoc su un singolo punto, senza passare da un DataFrame.

Uso:
    python execution/03_sensibilita_locale/sensitivity_analysis.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from cnav import Mission
from cnav.sensitivity.sensitivity_analysis_tools import (
    run_linear_sensitivity,
    run_sensitivity_3d_surface,
)

if __name__ == "__main__":

    # Tornado plot su un singolo punto operativo
    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")

    # Superficie 3D dell'elasticità di un singolo parametro (fan, H2 combustion)
    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)

    plt.show()
