Week 4: Model Evaluation, Validation, and Testing
A guide and runnable demo covering how to evaluate a machine learning model:
metrics, validation techniques, error analysis and a testing strategy for edge
cases and pitfalls.
What is in this repository
Path	Contents
`docs/pseudocode_and_flowcharts.md`	Flowcharts (Mermaid) and pseudocode for the evaluation pipeline
`src/evaluation_demo.py`	Runnable end-to-end demo on an imbalanced classification problem
`notebooks/evaluation_demo.ipynb`	The same demo as a notebook, with explanations between cells
`src/make_notebook.py`	Rebuilds the notebook from the script
`figures/`	Generated plots and the cross-validation results table
`requirements.txt`	Python dependencies
What the demo covers
Stratified train / validation / test split
Baseline and the accuracy paradox (97% accuracy, 0% recall)
Leakage-safe pipelines (imputer, scaler and model fitted together)
Stratified 5-fold cross-validation and nested cross-validation
Metrics: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, MCC, Brier score
Threshold selection on the validation set against a recall target
One-time test evaluation with bootstrap confidence intervals
Slice-based error analysis and review of missed positives
Edge-case tests: missing values, extreme inputs, noise, determinism, order invariance, directional checks
How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/evaluation_demo.py    # prints results, saves plots to figures/
jupyter notebook notebooks/evaluation_demo.ipynb
```
The script uses a synthetic dataset with a fixed random seed, so results are
reproducible. Expect about a minute of runtime.
Sample results (seed 42)
Check	Result
Majority-class baseline	Accuracy 0.965, recall 0.000
5-fold CV, Random Forest	PR-AUC 0.660 +/- 0.039
Nested CV, Random Forest	PR-AUC 0.666 +/- 0.035
Test set at chosen threshold	Recall 0.807, precision 0.292, ROC-AUC 0.921
Robustness checks	7 of 7 pass
Exact numbers can vary slightly across library versions.
Key lessons illustrated
Accuracy is misleading on imbalanced data; use PR-AUC, recall and precision.
Report mean and standard deviation from cross-validation, and confidence intervals on the test set.
Tune thresholds on validation data, never on the test set.
Overall scores can hide weak slices, so evaluate subgroups.
Test behaviour on bad inputs, not only on clean held-out data.
