"""
Riproduce la Fig. 4 dell'articolo: Electricity Intensity [MJ/(pax*nmi)] 
in funzione del range, a velocità di crociera fissata, 
un pannello per i velivoli a getto/fan e uno per quelli a elica.

Uso:
    python execution/01_riproduzione/plot_intensity_vs_range.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                   build_default_systems, build_energy_carriers, compute_intensity)

tech = TechAssumptions()
wtt = WellToTankEfficiencies()
systems = build_default_systems(build_energy_carriers(wtt))

ranges_nmi = np.geomspace(10, 10_000, 60)

fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

panels = [
    ("fan", 450, axes[0], "Jet/Fan at 450 kt"),
    ("propeller", 250, axes[1], "Propeller at 250 kt"),
]

for propulsor, speed_kt, ax, title in panels:
    for system in systems:
        intensities = []
        for r_nmi in ranges_nmi:
            mission = Mission(range_nmi=float(r_nmi), cruise_speed_kt=speed_kt, propulsor=propulsor)
            result = compute_intensity(mission, system, tech)
            intensities.append(result.intensity_MJ_per_pax_nmi)
        ax.plot(ranges_nmi, intensities, label=system.name, linewidth=3.0)

    ax.set_xscale("log")
    ax.set_ylim(0, 15)
    ax.set_ylabel("Electricity Intensity\n[MJ/(pax*nmi)]", fontsize=17)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, which="both", alpha=0.3)

axes[0].legend(loc="best", fontsize=16)
axes[-1].set_xlabel("Range [nmi]", fontsize=17)
fig.tight_layout()

out_path = Path(__file__).resolve().parent / "intensity_vs_range.png"
fig.savefig(out_path, dpi=150)
print(f"Grafico salvato in: {out_path}")
