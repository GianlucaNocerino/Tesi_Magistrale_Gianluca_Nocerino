"""
Esempio: screening multi-punto per individuare i parametri dominanti,
prima di passare alle mappe 2D complete (run_sensitivity_3d_surface).
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cnav.mission import Mission
from cnav.sensitivity_analysis_tools import (
    run_sensitivity_sweep,
    rank_dominant_parameters,
    run_sensitivity_3d_surface,
)

if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Griglia di punti operativi da campionare per lo screening.
    #    Rada apposta: qui serve solo capire chi e' dominante, non
    #    costruire ancora le mappe fini.
    # ---------------------------------------------------------
    ranges_nmi = [50, 200, 1000, 3000, 10000]
    speeds_kt = [250, 300, 350, 400, 500]
    propulsors = ["fan", "propeller"]

    missions = [
        Mission(range_nmi=r, cruise_speed_kt=v, propulsor=p)
        for r in ranges_nmi
        for v in speeds_kt
        for p in propulsors
    ]

    # ---------------------------------------------------------
    # 2. Sistemi propulsivi da confrontare.
    # ---------------------------------------------------------
    system_names = [
        "Battery-electric",
        "Hydrogen fuel cell",
        "Hydrogen combustion",
        "e-SAF combustion",
    ]

    # ---------------------------------------------------------
    # 3. Sweep: calcola le elasticita' su tutta la griglia,
    #    nessun grafico, solo il DataFrame tidy.
    # ---------------------------------------------------------
    df = run_sensitivity_sweep(missions, system_names, perturbation=0.01)
    print(f"Righe totali nel DataFrame: {len(df)}")

    # ---------------------------------------------------------
    # 4. Ranking dei dominanti, separato per sistema propulsivo
    #    (la fisica e' diversa tra batteria/H2/eSAF, quindi ha senso
    #    non aggregare tutto insieme).
    # ---------------------------------------------------------
    dominanti = rank_dominant_parameters(df, by=("system_name",), top_n=4)
    print("\nParametri dominanti per sistema (top 4):")
    print(dominanti.to_string(index=False))

    # ---------------------------------------------------------
    # 6. Solo ORA, per i parametri confermati dominanti, costruisce
    #    la mappa dell'elasticità al variare di range e velocità -
    #    per TUTTI i sistemi e sia fan che propeller. Con 4 sistemi x
    #    4 parametri x 2 propulsori sono 32 figure: le salviamo su
    #    disco invece di tenerle tutte a schermo con plt.show().
    # ---------------------------------------------------------
    range_grid = np.linspace(10, 10000, 50)
    speed_grid = np.linspace(200, 500, 50)

    out_dir = os.path.join(os.path.dirname(__file__), "surface_plots")
    os.makedirs(out_dir, exist_ok=True)

    for system_name in system_names:
        dominant_params = dominanti.loc[
            dominanti["system_name"] == system_name, "param_name"
        ].tolist()

        for param in dominant_params:
            for propulsor in ["fan", "propeller"]:
                run_sensitivity_3d_surface(
                    system_name, propulsor, param, range_grid, speed_grid
                )
                fname = f"{system_name}_{propulsor}_{param}.png".replace(" ", "_")
                plt.savefig(os.path.join(out_dir, fname), dpi=150)
                plt.close()

    print(f"\nMappe 2D salvate in: {out_dir}")