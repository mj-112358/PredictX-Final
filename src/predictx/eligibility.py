"""Model-specific applicability, essential inputs and empirical support indicators."""

import numpy as np


def inspect_inputs(panel, record, cfg, bundle):
    values = record.values
    features = cfg["features"]
    essential = cfg["essential"]
    missing = [k for k in essential if k not in values or not np.isfinite(values[k])]
    optional = [k for k in features if k not in essential and k not in values]
    labs = [k for k in features if k not in ["age", "sex"]]
    coverage = sum(k in values and np.isfinite(values[k]) for k in labs) / max(1, len(labs))
    hard = ["missing_essential:" + k for k in missing]
    warnings = []
    invalid = {
        k: v
        for k, v in record.rejected.items()
        if k in features or (k == "wbc" and "wbc_cells" in features)
    }
    for k, cause in invalid.items():
        warnings.append("measurement_withheld:" + k + ":" + cause)
    if optional:
        warnings.append("optional_inputs_imputed")
    if panel.age < cfg["min_age"]:
        hard.append("population_age_inapplicable")
    if cfg.get("sex") and panel.equation_sex != cfg["sex"]:
        hard.append("population_sex_inapplicable_or_unknown")
    if panel.acute_illness:
        hard.append("population_acute_illness_outside_study_scope")
    flags = [
        k
        for j, k in enumerate(features)
        if k in values
        and np.isfinite(values[k])
        and (values[k] < bundle.preprocessor.lo[j] or values[k] > bundle.preprocessor.hi[j])
    ]
    if flags:
        warnings.append("empirical_range_extrapolation")
    if coverage < cfg.get("training_coverage_q05", 0):
        warnings.append("coverage_below_training_q05")
    return {
        "eligible": not hard,
        "reasons": hard,
        "warning_codes": warnings,
        "missing_essential_inputs": missing,
        "missing_optional_inputs": optional,
        "invalid_or_withheld_measurements": invalid,
        "observed_feature_fraction": coverage,
        "out_of_range_features": flags,
        "domain_uncertainty": "limited_empirical_support"
        if flags or coverage < cfg.get("training_coverage_q05", 0)
        else "not_established_by_empirical_checks",
        "eligibility_policy": cfg.get("eligibility_policy", "v2_anchor_inputs_optional_imputation"),
        "reliability_gate": cfg["gate_status"],
    }
