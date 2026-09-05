"""
Sensitività globale.

La sequenza è quella della guideline: screening locale -> calibrazione
-> PDF -> sensibilità globale. Qui dentro, due passi:

  1. MORRIS (morris.py): screening a basso costo su tutti i fattori,
     r*(k+1) valutazioni. Ordina i fattori per influenza e segnala
     quali sono coinvolti in interazioni. Non quantifica.

  2. SOBOL (sobol.py): indici di varianza sul sottoinsieme scelto a mano dopo
     aver letto il Morris, con tutti gli altri fattori tenuti
     campionati e raggruppati in un unico fattore "resto", così la
     varianza totale resta quella vera e si può verificare che lo
     screening non abbia buttato via niente.

Lo spazio dei fattori (factors.py) non coincide con le 23 colonne di
theta: si campionano i quantili dei parametri indipendenti e un fattore
di gruppo per il blocco di calibrazione. Vedi il docstring di
factors.py per il perchè.

L'output è uno solo, la renewable electricity intensity, valutata su
un insieme di missioni rappresentative per ciascun sistema propulsivo
(outputs.py). I grafici a nube (mu*, sigma) con cui si filtrano i
fattori, uno per punto operativo più quello cumulativo, stanno in
morris.py accanto agli indici che disegnano
"""
from .factors import CALIBRATION_FACTOR, FactorSpace
from .morris import (
    aggregate_morris_cloud,
    elementary_effects,
    morris_cloud_table,
    morris_indices,
    morris_trajectories,
    plot_aggregate_morris_cloud,
    plot_all_morris_clouds,
    plot_morris_cloud,
    plot_morris_cloud_interactive,
    ranking_table,
    select_factors,
)
from .sobol import (
    RESIDUAL_GROUP,
    SobolDesign,
    compare_with_morris,
    plot_all_sobol_bars,
    plot_sobol_aggregate,
    plot_sobol_bars,
    plot_sobol_convergence,
    screening_check,
    sobol_convergence,
    sobol_design,
    sobol_indices,
    sobol_summary,
    sobol_table,
    tail_report,
)
from .outputs import (
    MISSION_LIBRARY,
    REPRESENTATIVE_MISSIONS,
    IntensityAt,
    default_outputs,
    evaluate_outputs,
    nan_report,
    outputs_from_missions,
    outputs_table,
)

__all__ = [
    "FactorSpace", "CALIBRATION_FACTOR",
    "IntensityAt", "MISSION_LIBRARY", "REPRESENTATIVE_MISSIONS",
    "outputs_from_missions", "outputs_table",
    "default_outputs", "evaluate_outputs", "nan_report",
    "morris_trajectories", "elementary_effects", "morris_indices",
    "ranking_table",
    "morris_cloud_table", "aggregate_morris_cloud", "select_factors",
    "plot_morris_cloud", "plot_all_morris_clouds",
    "plot_aggregate_morris_cloud", "plot_morris_cloud_interactive",
    "RESIDUAL_GROUP", "SobolDesign", "sobol_design", "sobol_indices",
    "sobol_summary", "screening_check", "sobol_table", "sobol_convergence",
    "compare_with_morris", "tail_report",
    "plot_sobol_bars", "plot_all_sobol_bars", "plot_sobol_aggregate",
    "plot_sobol_convergence",
]
