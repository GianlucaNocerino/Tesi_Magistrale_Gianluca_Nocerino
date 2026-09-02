"""
cnav.calibration - calibrazione del modello deterministico (cnav.model)
rispetto ai risultati del paper di riferimento. Vedi cost_functions.py
per i dettagli (funzioni di costo WLS, dati del paper, range massimo
fattibile per la batteria).
"""
from .cost_functions import (
    PAPER_DATA,
    SYSTEM_PARAMETERS,
    SHARED_PARAMETERS,
    DEFAULT_NONCONVERGENCE_PENALTY,
    fan_cost,
    propeller_cost,
    combined_cost,
    global_cost,
    system_cost,
    max_feasible_range_nmi,
)

from .deterministic_calibration import (
    FRACTION_PARAMETERS, CalibrationRun, default_bounds, manual_bounds,
    theta_to_tech_wtt, make_objective, latin_hypercube_starts,
    run_deterministic_calibration, summarize_multiple_minima,
)

from .calibration_uncertainty import (
    j_thr_from_margin,
    evaluate_J_over_sample,
    select_acceptable_region,
    CalibrationUncertaintyResult,
    run_calibration_uncertainty,
    accepted_correlation_matrix,
    sample_calibrated_theta,
)

__all__ = [
    "PAPER_DATA", "SYSTEM_PARAMETERS", "SHARED_PARAMETERS", "DEFAULT_NONCONVERGENCE_PENALTY",
    "fan_cost", "propeller_cost", "combined_cost", "global_cost", "system_cost", "max_feasible_range_nmi",
]
