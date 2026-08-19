"""
Stampa un grafico che, per ogni combinazione di
range e velocita' di crociera, mostra il sistema propulsivo più
efficiente (minor elecricity intensity).
Dopo aver effettuato la prima calibrazione...

Uso:
    python examples/plot_best_system_map_1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                   build_default_systems, build_energy_carriers, compute_intensity)

tech1 = TechAssumptions()
tech = tech1.with_changes(pax_weight_kg=85.02, fan_pressure_ratio=1.056, fan_scaling_mach_ref=0.7025, propeller_curve_peak_mach=0.4094, propeller_curve_rise_rate=17.24, propeller_curve_decay_width=0.01913)
wtt = WellToTankEfficiencies()

ranges_nmi = np.geomspace(10, 10_000, 45)
speeds_kt = np.linspace(180, 500, 40)

labels = []
for name in ("Battery-electric", "Hydrogen fuel cell", "Hydrogen combustion", "e-SAF combustion"):
    for propulsor in ("fan", "propeller"):
        labels.append(f"{name} ({propulsor})")
label_to_index = {label: i for i, label in enumerate(labels)}

best_index = np.full((len(speeds_kt), len(ranges_nmi)), -1, dtype=int)

for i, speed_kt in enumerate(speeds_kt):
    for j, r_nmi in enumerate(ranges_nmi):
        best_label, best_value = None, float("inf")
        for propulsor in ("fan", "propeller"):
            mission = Mission(range_nmi=float(r_nmi), cruise_speed_kt=float(speed_kt), propulsor=propulsor)
            systems = build_default_systems(build_energy_carriers(wtt))
            for system in systems:
                result = compute_intensity(mission, system, tech)
                value = result.intensity_MJ_per_pax_nmi
                if value == value and value < best_value:
                    best_value = value
                    best_label = f"{system.name} ({propulsor})"
        if best_label is not None:
            best_index[i, j] = label_to_index[best_label]

used_indices = sorted(set(best_index.flatten()) - {-1})
used_labels = [labels[k] for k in used_indices]
remap = {old: new for new, old in enumerate(used_indices)}
best_index_remapped = np.vectorize(lambda v: remap.get(v, -1))(best_index)

base_colors = {
    "Battery-electric": "#4C72B0",
    "Hydrogen fuel cell": "#DD8452",
    "Hydrogen combustion": "#55A868",
    "e-SAF combustion": "#C44E52",
}

def shade(hex_color, factor):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = (min(255, int(c * factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"

colors = []
for label in used_labels:
    name, propulsor = label.rsplit(" (", 1)
    base = base_colors[name]
    colors.append(base if "fan)" in label else shade(base, 1.35))

fig, ax = plt.subplots(figsize=(9, 6.5))
cmap = ListedColormap(colors)
ax.pcolormesh(ranges_nmi, speeds_kt, best_index_remapped, cmap=cmap,
              vmin=-0.5, vmax=len(used_labels) - 0.5, shading="auto")

ax.set_xscale("log")
ax.set_xlabel("Range [nmi]")
ax.set_ylabel("Velocità di crociera [kt]")
ax.set_title("Sistema propulsivo carbon-neutral più efficiente")

legend_handles = [Patch(color=colors[i], label=used_labels[i]) for i in range(len(used_labels))]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.92)

fig.tight_layout()
out_path = Path(__file__).resolve().parent / "best_system_map_1.png"
fig.savefig(out_path, dpi=150)
print(f"Grafico salvato in: {out_path}")