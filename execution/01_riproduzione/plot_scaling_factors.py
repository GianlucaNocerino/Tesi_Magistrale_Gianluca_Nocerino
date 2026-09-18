"""
Genera il grafico del fattore di scala dell'efficienza propulsiva
(fan ed propeller) in funzione del numero di Mach

Uso:
    python execution/01_riproduzione/plot_scaling_factors.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import TechAssumptions
from cnav import propulsive_efficiency as pe

tech = TechAssumptions()
machs = np.linspace(0.0, 1.0, 400)

fan_vals = [pe.fan_efficiency_scaling(m, tech.fan_pressure_ratio, tech.fan_scaling_mach_ref) for m in machs]
prop_vals = [pe.propeller_efficiency_scaling(
    m, tech.propeller_curve_peak_mach, tech.propeller_curve_rise_rate,
    tech.propeller_curve_decay_width) for m in machs]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(machs, fan_vals, color="crimson", lw=3,
        label=f"Fan")
ax.plot(machs, prop_vals, color="steelblue", lw=3, linestyle="--",
        label="Propeller")

ax.set_xlabel("Flight Mach", fontsize=16)
ax.set_ylabel("Propulsive Efficiency's Scaling Factor", fontsize=16)
ax.set_title("Scaling Factor: fan vs propeller", fontsize=18, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.legend(loc="best", fontsize=14, framealpha=0.5)
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = Path(__file__).resolve().parent / "scaling_factors.png"
fig.savefig(out_path, dpi=150)
print(f"Grafico salvato in: {out_path}")