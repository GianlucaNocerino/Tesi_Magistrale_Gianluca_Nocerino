"""
Sensibilità globale (indici di Sobol) della renewable electricity
intensity rispetto ai parametri tecnologici incerti.

Niente screening: i parametri da analizzare singolarmente li sceglie
l'analista (SELECTED_FACTORS nello script). Tutti gli altri restano
campionati dalle loro PDF e formano il gruppo "others", così la varianza
al denominatore è quella vera e S_T[others] misura quanto pesa
l'incertezza non tecnologica.

Moduli:
  factors.py      lo spazio dei fattori (quantili indipendenti + blocco
                  di calibrazione)
  outputs.py      la griglia range-velocità x architetture, la
                  valutazione del modello, la pseudo-architettura E_best
  sobol.py        disegno di Saltelli, indici con bootstrap condiviso
  aggregation.py  come si aggregano indici di output diversi (media
                  pesata con la varianza = indice generalizzato), mappe
                  del parametro più influente
  plots.py        barre aggregate, mappe, convergenza
"""
from .factors import CALIBRATION_FACTOR, FactorSpace
from .outputs import (
    BEST_SYSTEM,
    MISSION_LIBRARY,
    PROPULSORS,
    REPRESENTATIVE_MISSIONS,
    SYSTEMS,
    IntensityAt,
    append_best_system,
    architecture_label,
    default_outputs,
    evaluate_outputs,
    grid_outputs,
    nan_report,
    outputs_from_missions,
    outputs_table,
    prescreen_outputs,
)
from .sobol import (
    RESIDUAL_GROUP,
    SobolDesign,
    SobolField,
    residual_report,
    sobol_design,
    sobol_field,
    sobol_table,
    tail_report,
)
from .aggregation import (
    WEIGHT_SCHEMES,
    aggregate_indices,
    best_probability,
    best_system_map,
    define_groups,
    influence_map,
    sobol_convergence,
)
from .plots import (
    ARCHITECTURE_COLORS,
    FACTOR_COLORS,
    FACTOR_LABELS,
    factor_label,
    plot_best_system_map,
    plot_influence_map,
    plot_sobol_aggregate,
    plot_sobol_convergence,
)

__all__ = [
    "FactorSpace", "CALIBRATION_FACTOR",
    "SYSTEMS", "PROPULSORS", "BEST_SYSTEM", "IntensityAt", "MISSION_LIBRARY",
    "REPRESENTATIVE_MISSIONS", "architecture_label", "grid_outputs",
    "outputs_from_missions", "default_outputs", "outputs_table", "evaluate_outputs",
    "prescreen_outputs", "append_best_system", "nan_report",
    "RESIDUAL_GROUP", "SobolDesign", "SobolField", "sobol_design", "sobol_field",
    "tail_report", "residual_report", "sobol_table",
    "WEIGHT_SCHEMES", "define_groups", "best_probability", "aggregate_indices",
    "influence_map", "best_system_map", "sobol_convergence",
    "FACTOR_LABELS", "FACTOR_COLORS", "ARCHITECTURE_COLORS", "factor_label",
    "plot_sobol_aggregate", "plot_influence_map", "plot_best_system_map",
    "plot_sobol_convergence",
]
