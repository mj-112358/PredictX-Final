# Experimental methods

## Data and partitions

`data/raw/source_manifest.json` hashes the retained public source files. The acquisition manifests record original download URLs and the Senegal dataset DOI. ILPD CSV bytes originate from the UCI archive identified in `manifest.json`. The source tables retain anonymized research identifiers.

`data.py` assembles UCI CKD and ILPD covariates, removes exact feature/label duplicates, and constructs stable row identifiers. NHANES tables join on SEQN with one-to-one validation. The adult analytic population includes records with an observed laboratory feature and at least one source outcome. Four questionnaire outcomes define the original NHANES analytic population and partition; the deployed system uses its diabetes and cardiovascular classifiers. `breast_data.py` loads the 40 TNBC, 40 hormone-dependent and 40 control records, checks participant identifiers, and applies the same CBC consistency rule used at inference.

The fixed seed is 2026. `scripts/experiment.py:split` first allocates 40% training and 60% remainder, then divides the remainder into 20% validation, 20% calibration and 20% test using sequential stratified splits. Stratification uses the first source endpoint with missing labels represented as a separate stratum. Serialized partition identifiers provide the exact membership for every retained fit.

Median imputation and standard scaling are fitted on training covariates. Training quantiles 0.005 and 0.995 provide empirical bounds. Standardized values are clipped to [-10, 10], followed by observed-value indicators and their per-record mean. Predictors use float32 transformed inputs.

## Independent fitting and calibration

The original candidate families are logistic regression, histogram gradient boosting and independently trained MLPs. Hyperparameters are selected by validation loss. Each disease output has a logistic sigmoid calibrator fitted on its calibration partition. Calibrated validation log loss, with Brier score as a tie-breaker, selects the retained source model. `results/independent/architecture_selection.csv` records the comparisons; `results/models` retains the corresponding independently fitted candidates and partition metadata.

The serialized MLP class `predictx.models.SharedNetwork` retains the serialization identity required by the liver checkpoint. Its executable implementation is a single-output perceptron with two 32-unit hidden layers. The independent training function masks missing outcome labels during loss calculation.

Additional liver and cardiovascular comparisons use three outer training folds (seed 2027), two inner folds for outer-fold selection and three-fold full-training selection (seed 2026). Candidate families include logistic regression, boosting, random forests and a soft-voting ensemble. Preprocessing is fitted within each training fold. Final candidates use the original calibration partition and calibrated validation loss. Replacement requires a relative validation-loss reduction greater than 1%; both retained baseline models satisfy the recorded selection procedure. Candidate configurations, validation predictions, held-out predictions and checkpoints remain in `results/refinement`.

Calibration reconstruction is checked against the mean regularized logistic-loss gradient at the stored L-BFGS convergence tolerance and the existing inference score-parity contract (relative 1e-6, absolute 1e-7). The audit records coefficient, intercept and probability differences. This accommodates platform-dependent float32 neural scores while retaining artifact digests and saved-prediction comparisons. Candidate validation losses use the retained evaluation precision (relative 1e-6, absolute 1e-7), with exact validation confusion counts at the reconstructed development-sensitivity cutoff and an unchanged winning architecture. The reconstructed cutoff must match the stored value at score precision; absolute loss differences, cutoff differences and stored-cutoff boundary changes are recorded.

## Operating points and evaluation

Validation records satisfying the input contract determine Youden, sensitivity-90%, specificity-80% and illustrative FN:FP cost 4:1 operating points. The registry uses the validation Youden threshold. Threshold reconstruction uses the score-parity precision and verifies the retained validation confusion counts exactly. Binary reporting requires validation sensitivity ≥0.8 and specificity ≥0.6; liver and breast assessments use ranking mode under this rule.

The component metrics use all labeled test records. Runtime coverage separately counts eligible records, scored true/false positives and negatives, and abstained positive/negative records. `scripts/evaluate.py` reconstructs canonical source records and verifies every eligible test score against the inference function. Source assay quality, collection timing and acute-illness metadata are accepted assumptions in this retrospective reconstruction. CKD canonical validity processing changes a few component scores slightly; the retained canonical reference predictions and their measured metrics define the patent comparison.

AUROC confidence intervals use 500 paired IID resamples of the held-out `(label, score)` records with NumPy seed 2026. Single-class draws are omitted. The 2.5th and 97.5th percentiles form the interval. Prediction functions remain fixed during resampling. Thus the CKD interval 1.000–1.000 describes conditional discrimination in the 80 evaluated records; it provides no estimate of performance variation across newly trained models or external populations. Brier score and logarithmic loss use the retained sigmoid-calibrated scores.

These previously inspected source-cohort test results support retrospective classification comparisons. Their calibration applies to the corresponding source endpoints and sampling designs. External population screening and prospective clinical performance require independent evaluation.

## Computational experiments

`benchmark_refined.py` evaluates seven arrangements with matched synthetic records and real fitted predictors: parallel workers, sequential workers, feature-minimal independent pipelines, full independent pipelines, integrated execution, integrated execution with transformation-cache reuse disabled, and vectorized independent execution with shared ingestion.

CPU execution uses one numerical thread, five warm-up batches, eight randomized rounds and three repetitions per round at batch sizes 1, 8 and 32. Every measured score array is compared against integrated execution. Counters record canonical normalization calls, projections, transforms and cache hits. Five full input paths perform 190 canonical operations; integrated input performs 38. Exact diabetes/cardiovascular fitted-state compatibility reduces transformations from five to four.

Fresh-interpreter startup uses three repetitions with randomized arrangement order. Measurements include imports, model loading and process setup with the operating-system file cache retained. Memory records peak RSS for the integrated process and summed independently measured worker RSS peaks; these peaks have process-specific timing and shared-page accounting. Linux PSS is recorded when available. `benchmark_model_loading.py` separately measures integrity verification and deserialization with warmed imports and file caches. Local multiprocessing IPC is included in worker execution measurements.

Frozen patent measurements remain in `results/refinement`; each reproduction writes its own environment, raw samples, summaries and model hashes under `results/reproduced`.
