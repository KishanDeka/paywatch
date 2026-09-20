# Model card

## Intended use

An educational, offline-evaluated fraud triage portfolio. The service records pass/review
recommendations; it is not connected to payment processing. Intended reviewers should inspect the
model mathematics, leakage controls, export checks, failure handling, and trade-offs.

## Training data and outputs

Supported real input: the ULB/Kaggle anonymized credit-card transaction dataset. Features are
28 supplied PCA dimensions plus transformed amount and interarrival time. The supplied demonstration
is artificial, has much higher fraud prevalence than the real dataset, and encodes a mixture of
strong anomalies and subtle labeled cases. Its regular injection schedule is not realistic.

The LSTM estimates reconstruction deviation. XGBoost estimates a calibrated probability only for
routed events in serving. No protected attributes or demographic labels are provided, so group
fairness cannot be established. Anomaly is not synonymous with fraud.

## Evaluation

Report full-test average precision, ROC-AUC, threshold precision/recall, confusion matrix,
gate recall, classifier fraction, review fraction, captured fraudulent amount, missed amount,
and illustrative cost. `average_precision` is sklearn's non-interpolated average precision,
not trapezoidal area under the precision–recall curve. Do not swap the labels without explanation.

For the cascade ranking metric only, bypassed events receive score zero. That convention tests
ranking of the operationally available scores but those zeros are not probabilities. Runtime
bypasses remain null. Brier score is reported for the all-event calibrated XGBoost output only.

Recall uncertainty uses a moving-segment-style non-overlapping block bootstrap (100 consecutive
rows per block, sampled with replacement, 500 repeats). It partially preserves short-range
dependence. It is descriptive, sensitive to block size, and weak with few fraud-containing blocks.
It does not certify a 98% recall guarantee or future distribution performance.

The calibration and policy sets are development data. Only the final chronological test is held
out. Do not select a different configuration after seeing its test score without marking that
period as development data and reserving a new final test.

## Limitations and failure modes

- No entity ID: global-stream context may be weak or spurious.
- Only a short historical dataset: no demonstrated long-horizon drift robustness.
- Normal-only windows use labels assumed known in offline historical training; real delayed labels
  would require a lag-aware training cutoff.
- Anomalies can be legitimate; familiar-looking fraud can bypass the gate.
- Previous anomalous rows influence later reconstruction, potentially causing alert bursts.
- Synthetic or test-specific performance is not a business deployment claim.
- The model is not validated for automatic transaction blocking, credit decisions, or identity checks.
- The three largest standardized feature deviations in an audit are diagnostics, not SHAP values,
  causal explanations, or interpretable customer behavior. Anonymous PCA components limit meaning.
- Point-estimate gate selection is unstable with few policy positives; the included synthetic run
  demonstrates this failure. More data and temporal folds matter more than extra infrastructure.

## Reproducibility and release

Each run records input checksum, feature order, split ranges/counts, seed, configuration, package
versions, model version, and artifact checksums. Re-running on another CPU or library stack can
produce small numerical differences. The README's original performance numbers are omitted as
unverified. See `reports/VALIDATION.md` for exactly what was executed here.

The code's release check is a configurable offline check, not regulatory approval or proof of
production safety. Its default recall/cost limits are example project criteria. Real deployment
requires an agreed review process, delayed-label evaluation, capacity constraints, security, and
measured fault recovery.
