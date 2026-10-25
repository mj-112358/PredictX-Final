# Patent implementation map

The accompanying [provisional specification](PredictX_provisional_specification.docx) defines the retained architecture and reported experiments.

| Patent component | Implementation | Verification or evidence |
|---|---|---|
| §5.1 Typed observations and canonical units | `schema.py`, `Measurement`, `LabPanel`, `canonical_value` | Unit, specimen, time-domain and plausibility tests |
| §5.1 Source fingerprint and canonical snapshot | `LabPanel._fingerprint`, `canonical_snapshot` | Mutation tests include observation-source metadata |
| §5.2 Quality and feature dependencies | `processing.py:cross_field_qc`; `biomarkers.py` | Shared source/runtime CBC QC and recipe tests |
| §5.3 Independent models and registry | `runtime.py`, `inference.py:load_model`, `configs/model_registry.json` | Five retained SHA-256 checks and schema/endpoint tests |
| §5.3 Fitted preprocessing and calibration | `FittedFeatures`, `DiseaseModel` | Training statistics and per-record score parity |
| §5.4 Compatibility controller | `compatibility.py`, `processing.py:project/transform` | Complete-state signature perturbations, engine key and request-isolation tests |
| §5.5 Applicability and reporting | `eligibility.py`, `inference.py` | Per-model anchors, population checks, quality codes and five outputs |
| §5.6 API and interface | `api.py`, `demo/app.py` | Single/batch API tests and Streamlit AppTest |
| §6 Disease-specific embodiments | `data.py`, `breast_data.py`, `results/deployed` | Public source hashes, partitions and fitted artifacts |
| §8 Evaluation and model comparison | `scripts/evaluate.py`, `experiment.py`, `refine_models.py` | Retained predictions and recomputed metrics/figures |
| §9 Efficiency and resource use | `scripts/benchmark_refined.py`, `benchmark_model_loading.py` | Raw randomized samples, operation counts, startup and RSS |

The existing flat module layout follows these boundaries while preserving model serialization paths. Biochemical numerical recipes are shared by cohort feature generation and laboratory inference. The compatibility digest uses the complete fitted `FittedFeatures` object: ordered columns, sklearn imputer/scaler state, indicator policy, and empirical low/high bounds. The feature-engine version forms the other cache-key component. Read-only transformed arrays and per-record cache dictionaries enforce request isolation.

Reconstruction adds the disclosed `cells/mm³ → 10⁹/L` concentration conversion, equivalent ASCII unit forms and Unicode micro-unit aliases. The snapshot fingerprint now includes reference intervals and laboratory-source metadata. Original model files and registry checksums remain intact.
