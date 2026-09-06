"""
cnav.uncertainty: definizione delle PDF
dei parametri, propagazione dell'incertezza attraverso il modello
deterministico, e mappa probabilistica delle tecnologie.

Tre moduli, uno per fase, da usare in quest'ordine:

    distributions.py    PDF dichiarate + correlazioni fisiche +
                                campionamento LHS + fusione con Theta_acc
                                (i campioni di calibrazione della Fase 6,
                                prodotti da cnav.calibration)
    propagation.py      esecuzione del modello su ogni campione,
                                su tutta la griglia (R, V)
    technology_map.py   P_j(R, V), regioni robuste e
                                decision-uncertain, dispersione dei confini

Il taglio è lo stesso degli altri sottopacchetti: cnav.uncertainty
lavora SOPRA cnav.model senza modificarlo, e non contiene nessuna
formula del modello deterministico.

Vedi examples/uncertainty_propagation_execution.py per il flusso
completo.
"""
from .distributions import (
    ParameterSpec,
    triangular,
    scaled_beta,
    reflected_lognormal,
    build_default_specs,
    specs_table,
    CopulaPair,
    DerivedParameter,
    DerivedFromMany,
    induced_spec,
    induced_spec_many,
    CorrelationModel,
    build_default_correlations,
    shared_factor_rho,
    independent_parameter_names,
    sample_literature_parameters,
    load_theta_acc,
    assemble_theta,
)
from .model_form_uncertainty import (
    FORM_SPECIFIC_PARAMETERS,
    assemble_theta_for_form,
    common_parameter_names,
    correlations_for_specs,
    parameter_membership_table,
    specs_for_form,
)
from .propagation import (
    TECHNOLOGIES,
    PROPULSORS,
    FlightGrid,
    technology_labels,
    theta_row_to_tech_wtt,
    evaluate_sample,
    PropagationResult,
    run_propagation,
)
from .technology_map import (
    ProbabilityMap,
    probability_map,
    aggregate_to_technologies,
    classify_regions,
    robust_area_fraction,
    intensity_percentiles,
    feasibility_probability,
    boundary_statistics,
    adjacent_label_pairs,
    plot_probability_map,
    plot_probability_field,
    plot_intensity_band,
    plot_boundary_dispersion,
)
 
__all__ = [
    "ParameterSpec", "triangular", "scaled_beta", "reflected_lognormal",
    "build_default_specs",
    "specs_table", "CopulaPair", "CorrelationModel",
    "DerivedParameter", "DerivedFromMany",
    "induced_spec", "induced_spec_many",
    "build_default_correlations", "shared_factor_rho",
    "sample_literature_parameters", "independent_parameter_names", 
    "load_theta_acc", "assemble_theta",
    "TECHNOLOGIES", "PROPULSORS", "FlightGrid", "technology_labels",
    "theta_row_to_tech_wtt", "evaluate_sample", "PropagationResult",
    "run_propagation",
    "ProbabilityMap", "probability_map", "aggregate_to_technologies",
    "classify_regions", "robust_area_fraction", "intensity_percentiles",
    "feasibility_probability", "boundary_statistics", "adjacent_label_pairs",
    "plot_probability_map", "plot_probability_field",
    "plot_intensity_band", "plot_boundary_dispersion",
    "FORM_SPECIFIC_PARAMETERS", "common_parameter_names",
    "specs_for_form", "correlations_for_specs", "assemble_theta_for_form",
    "parameter_membership_table",
]