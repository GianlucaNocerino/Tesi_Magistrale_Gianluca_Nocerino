"""
Genera il grafico del fattore di scala dell'efficienza propulsiva
(fan ed propeller) in funzione del numero di Mach.
Dopo aver effettuato la terza calibrazione...

Uso:
    python execution/01_riproduzione/plot_scaling_factors_2.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import TechAssumptions
from cnav import propulsive_efficiency as pe

tech2 = TechAssumptions()
tech = tech2.with_changes(pax_weight_kg=75, fan_pressure_ratio=1.3, fan_scaling_mach_ref=0.78, propeller_curve_peak_mach=0.4899, propeller_curve_rise_rate=7.504, propeller_curve_decay_width=0.03354)
machs = np.linspace(0.0, 1.0, 400)

fan_vals = [pe.fan_efficiency_scaling(m, tech.fan_pressure_ratio, tech.fan_scaling_mach_ref) for m in machs]
prop_vals = [pe.propeller_efficiency_scaling(
    m, tech.propeller_curve_peak_mach, tech.propeller_curve_rise_rate,
    tech.propeller_curve_decay_width) for m in machs]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(machs, fan_vals, color="crimson", lw=2.5,
        label=f"Fan (Michel [55], pi_fan={tech.fan_pressure_ratio})")
ax.plot(machs, prop_vals, color="steelblue", lw=2.5, linestyle="--",
        label="Propeller")

ax.set_xlabel("Numero di Mach di volo")
ax.set_ylabel("Fattore di scala dell'efficienza propulsiva")
ax.set_title("Fattore di scala: fan vs propeller")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = Path(__file__).resolve().parent / "scaling_factors_2.png"
fig.savefig(out_path, dpi=150)
print(f"Grafico salvato in: {out_path}")