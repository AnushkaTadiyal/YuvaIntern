import os, time, pickle, warnings
import numpy as np, pandas as pd, joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate,
    cross_val_predict, cross_val_score, RandomizedSearchCV, validation_curve, learning_curve)
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss, precision_recall_curve)
warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)
_cur = [None]
def step(name):
    _cur[0] = name; open(f"logs/{name}.txt", "w").close()
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    open(f"logs/{_cur[0]}.txt", "a").write(s + "\n")
R = {}   # results for plotting
SEED = 42

#@@ STEP 1_data
step("1_data")
rng = np.random.default_rng(SEED)
n = 12000
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
internet = rng.choice(["DSL", "Fiber", "None"], n, p=[.35, .45, .20])
payment = rng.choice(["Card", "Bank transfer", "E-check", "Mailed check"], n, p=[.3, .3, .25, .15])
tenure = np.clip(rng.gamma(2.0, 14, n), 1, 72).round()
monthly = np.clip(rng.normal(65, 25, n) + 15*(internet == "Fiber") - 30*(internet == "None"), 18, 120)
total = tenure * monthly * rng.normal(1, 0.05, n)
support = rng.poisson(1.2 + 0.8*(internet == "Fiber"))
late = rng.poisson(0.4 + 0.5*(payment == "E-check"))
age = rng.integers(18, 81, n)
products = rng.integers(1, 6, n)
usage = rng.lognormal(3.0, 0.8, n)
z = (1.3*(contract == "Month-to-month") - 0.9*(contract == "Two year") - 0.035*tenure
     + 0.018*(monthly - 65) + 0.30*support + 0.35*late - 0.22*products
     + 0.8*((internet == "Fiber") & (monthly > 85))
     + 0.9*((contract == "Month-to-month") & (tenure < 6)) + rng.normal(0, 0.5, n))
lo, hi = -10.0, 10.0
for _ in range(60):                     # find intercept giving ~20% churn
    mid = (lo + hi) / 2
    if (1/(1 + np.exp(-(z + mid)))).mean() > 0.20: hi = mid
    else: lo = mid
y = pd.Series((rng.random(n) < 1/(1 + np.exp(-(z + mid)))).astype(int), name="churn")
X = pd.DataFrame({"tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
    "support_calls_90d": support, "late_payments": late, "age": age, "num_products": products,
    "data_usage_gb": usage, "contract": contract, "internet_service": internet,
    "payment_method": payment})
X.loc[rng.random(n) < 0.05, "data_usage_gb"] = np.nan      # inject realistic missingness
X.loc[rng.random(n) < 0.03, "monthly_charges"] = np.nan
P("Dataset shape:", X.shape, "| churn rate: %.1f%%" % (100*y.mean()))
P("Missing values (%):", (100*X.isna().mean()[lambda s: s > 0]).round(1).to_dict())

#@@ STEP 2_split
step("2_split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
P("Train: %d rows, churn %.1f%%" % (len(X_train), 100*y_train.mean()))
P("Test : %d rows, churn %.1f%%" % (len(X_test), 100*y_test.mean()))

num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d",
            "late_payments", "age", "num_products", "data_usage_gb"]
cat_cols = ["contract", "internet_service", "payment_method"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
def make(clf): return Pipeline([("prep", preprocess), ("clf", clf)])
SC = {"pr": "average_precision", "roc": "roc_auc", "f1": "f1"}

#@@ STEP 3_baseline_vs_models
step("3_models")
dummy = make(DummyClassifier(strategy="prior")).fit(X_train, y_train)
P("Dummy (always predicts 'stay'): test accuracy = %.3f, recall = %.3f"
  % (accuracy_score(y_test, dummy.predict(X_test)), recall_score(y_test, dummy.predict(X_test))))
cands = {
  "Baseline (prior)":   DummyClassifier(strategy="prior"),
  "Logistic regression": LogisticRegression(max_iter=2000),
  "Random forest":      RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
  "Gradient boosting":  HistGradientBoostingClassifier(random_state=SEED),
}
P("%-22s %-16s %-16s %-16s %s" % ("Model", "PR-AUC", "ROC-AUC", "F1@0.5", "Fit time/fold"))
R["cv_models"] = {}
for name, clf in cands.items():
    t = time.time()
    s = cross_validate(make(clf), X_train, y_train, cv=cv, scoring=SC)
    R["cv_models"][name] = {k: (s["test_"+k].mean(), s["test_"+k].std()) for k in SC}
    P("%-22s %.3f +/- %.3f   %.3f +/- %.3f   %.3f +/- %.3f   %.1fs" % (name,
      *R["cv_models"][name]["pr"], *R["cv_models"][name]["roc"], *R["cv_models"][name]["f1"],
      (time.time()-t)/5))

#@@ STEP 4_imbalance
step("4_imbalance")
P("%-34s %-10s %-10s %-10s %s" % ("Gradient boosting variant", "PR-AUC", "Precision", "Recall", "F1"))
R["imb"] = {}
for name, cw in [("No class weights", None), ("class_weight='balanced'", "balanced")]:
    s = cross_validate(make(HistGradientBoostingClassifier(class_weight=cw, random_state=SEED)),
        X_train, y_train, cv=cv, scoring={"pr": "average_precision", "p": "precision", "r": "recall", "f1": "f1"})
    R["imb"][name] = {k: s["test_"+k].mean() for k in ["pr", "p", "r", "f1"]}
    P("%-34s %-10.3f %-10.3f %-10.3f %.3f" % (name, *[R["imb"][name][k] for k in ["pr", "p", "r", "f1"]]))

#@@ STEP 5_regularisation
step("5_logreg_C")
P("%-10s %-14s %s" % ("C", "CV PR-AUC", "(smaller C = stronger L2 regularisation)"))
R["logreg_C"] = []
for C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    s = cross_val_score(make(LogisticRegression(C=C, max_iter=2000)), X_train, y_train, cv=cv, scoring="average_precision")
    R["logreg_C"].append((C, s.mean(), s.std())); P("%-10g %.3f +/- %.3f" % (C, s.mean(), s.std()))

#@@ STEP 6_validation_curve
step("6_valcurve")
leaves = [3, 7, 15, 31, 63, 127, 255]
tr, va = validation_curve(make(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
    early_stopping=False, random_state=SEED)), X_train, y_train, param_name="clf__max_leaf_nodes",
    param_range=leaves, cv=cv, scoring="average_precision")
R["valcurve"] = (leaves, tr, va)
P("%-12s %-12s %-12s %s" % ("max_leaf_nodes", "Train PR-AUC", "CV PR-AUC", "Gap"))
for l, a, b in zip(leaves, tr.mean(1), va.mean(1)): P("%-12d %-12.3f %-12.3f %.3f" % (l, a, b, a-b))

#@@ STEP 7_learning_curve
step("7_learning")
sizes, tr, va = learning_curve(make(HistGradientBoostingClassifier(random_state=SEED)), X_train, y_train,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=cv, scoring="average_precision", random_state=SEED)
R["learning"] = (sizes, tr, va)
P("%-12s %-12s %s" % ("Train rows", "Train PR-AUC", "CV PR-AUC"))
for s_, a, b in zip(sizes, tr.mean(1), va.mean(1)): P("%-12d %-12.3f %.3f" % (s_, a, b))

#@@ STEP 8_tuning
step("8_tuning")
param_dist = {
    "clf__learning_rate": loguniform(0.01, 0.3), "clf__max_iter": randint(100, 400),
    "clf__max_leaf_nodes": randint(4, 48), "clf__min_samples_leaf": randint(10, 120),
    "clf__l2_regularization": loguniform(1e-3, 10), "clf__max_features": uniform(0.4, 0.6)}
search = RandomizedSearchCV(make(HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)),
    param_dist, n_iter=20, cv=cv, scoring="average_precision", random_state=SEED, refit=True)
t = time.time(); search.fit(X_train, y_train)
P("Random search: 20 configs x 5 folds = 100 fits in %.0f s" % (time.time() - t))
P("Default model CV PR-AUC : %.3f" % R["cv_models"]["Gradient boosting"]["pr"][0])
P("Tuned model   CV PR-AUC : %.3f" % search.best_score_)
P("Best parameters:", {k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, float) else int(v))
                       for k, v in search.best_params_.items()})
res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)
P("\nTop-5 configurations:")
P("%-5s %-14s %-9s %-9s %-9s %-9s %s" % ("Rank", "CV PR-AUC", "lr", "n_iter", "leaves", "min_leaf", "l2"))
for _, r_ in res.iterrows():
    P("%-5d %.3f +/- %.3f  %-9.3f %-9d %-9d %-9d %.3f" % (r_.rank_test_score, r_.mean_test_score, r_.std_test_score,
      r_.param_clf__learning_rate, r_.param_clf__max_iter, r_.param_clf__max_leaf_nodes,
      r_.param_clf__min_samples_leaf, r_.param_clf__l2_regularization))
allscores = pd.DataFrame(search.cv_results_).mean_test_score
P("\nAll 20 configs: min %.3f, median %.3f, max %.3f" % (allscores.min(), allscores.median(), allscores.max()))
R["search_scores"] = allscores.values
best = search.best_estimator_

#@@ STEP 9_threshold
step("9_threshold")
oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
COST_FN, COST_FP = 300, 40       # lost margin of a missed churner vs cost of a wasted retention offer
grid = np.linspace(0.02, 0.95, 94)
def at(th, p=oof, yy=y_train):
    pred = (p >= th).astype(int); tn, fp, fn, tp = confusion_matrix(yy, pred).ravel()
    return dict(th=th, prec=precision_score(yy, pred), rec=recall_score(yy, pred), f1=f1_score(yy, pred),
                cost=(COST_FN*fn + COST_FP*fp)/len(yy)*1000)
f1_th = grid[np.argmax([at(t_)["f1"] for t_ in grid])]
cost_th = grid[np.argmin([at(t_)["cost"] for t_ in grid])]
P("Out-of-fold predictions on the training set (cost: FN=$%d, FP=$%d)" % (COST_FN, COST_FP))
P("%-22s %-10s %-10s %-10s %-8s %s" % ("Rule", "Threshold", "Precision", "Recall", "F1", "Cost per 1,000 customers"))
R["thr_rows"] = {}
for name, th in [("Default 0.50", 0.50), ("F1-optimal", f1_th), ("Cost-optimal", cost_th)]:
    a = at(th); R["thr_rows"][name] = a
    P("%-22s %-10.2f %-10.3f %-10.3f %-8.3f $%s" % (name, th, a["prec"], a["rec"], a["f1"], format(round(a["cost"]), ",")))
R["oof"] = oof; R["thr"] = dict(f1=f1_th, cost=cost_th)

#@@ STEP 10_test
step("10_test")
test_models = {n_: make(c).fit(X_train, y_train) for n_, c in cands.items() if n_ != "Baseline (prior)"}
test_models["Gradient boosting (tuned)"] = best
test_models["Baseline (prior)"] = dummy
P("%-28s %-10s %s" % ("Model (held-out test set)", "PR-AUC", "ROC-AUC"))
R["test_scores"], R["test_proba"] = {}, {}
for name in ["Baseline (prior)", "Logistic regression", "Random forest", "Gradient boosting", "Gradient boosting (tuned)"]:
    pr_ = test_models[name].predict_proba(X_test)[:, 1]; R["test_proba"][name] = pr_
    R["test_scores"][name] = (average_precision_score(y_test, pr_), roc_auc_score(y_test, pr_))
    P("%-28s %-10.3f %.3f" % (name, *R["test_scores"][name]))
proba = R["test_proba"]["Gradient boosting (tuned)"]
pred = (proba >= f1_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
P("\nFinal model at F1-optimal threshold = %.2f" % f1_th)
P("Confusion matrix: TN=%d FP=%d FN=%d TP=%d" % (tn, fp, fn, tp))
P("Accuracy %.3f | Precision %.3f | Recall %.3f | F1 %.3f" % (accuracy_score(y_test, pred),
  precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)))
R["cm"] = (tn, fp, fn, tp)
yt = y_test.values; brng = np.random.default_rng(SEED); bs_pr, bs_f1 = [], []
for _ in range(1000):
    idx = brng.integers(0, len(yt), len(yt))
    bs_pr.append(average_precision_score(yt[idx], proba[idx])); bs_f1.append(f1_score(yt[idx], pred[idx]))
P("95%% bootstrap CI  PR-AUC: [%.3f, %.3f]   F1: [%.3f, %.3f]" % (*np.percentile(bs_pr, [2.5, 97.5]), *np.percentile(bs_f1, [2.5, 97.5])))
order = np.argsort(-proba); top = order[:len(order)//10]
P("Top-decile lift: %.2fx (top 10%% of scores contain %.0f%% of all churners)" %
  (yt[top].mean()/yt.mean(), 100*yt[top].sum()/yt.sum()))
a = at(cost_th, proba, y_test); b0 = at(0.5, proba, y_test)
P("Expected cost per 1,000 customers on test: threshold 0.50 -> $%s ; cost-optimal %.2f -> $%s ; no model -> $%s" %
  (format(round(b0["cost"]), ","), cost_th, format(round(a["cost"]), ","), format(round(COST_FN*yt.mean()*1000), ",")))

#@@ STEP 11_calibration
step("11_calibration")
def ece(yy, p, bins=10):
    edges = np.linspace(0, 1, bins+1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins-1); e = 0
    for b_ in range(bins):
        m = idx == b_
        if m.any(): e += m.mean() * abs(yy[m].mean() - p[m].mean())
    return e
bal = clone(best).set_params(clf__class_weight="balanced").fit(X_train, y_train)
iso = CalibratedClassifierCV(clone(best), method="isotonic", cv=5).fit(X_train, y_train)
P("%-36s %-12s %s" % ("Model (test set)", "Brier score", "ECE (10 bins)"))
R["calib"] = {}
for name, m in [("Tuned (no class weights)", best), ("Tuned + class_weight='balanced'", bal), ("Tuned + isotonic calibration", iso)]:
    p_ = m.predict_proba(X_test)[:, 1]; R["calib"][name] = (p_, brier_score_loss(yt, p_), ece(yt, p_))
    P("%-36s %-12.4f %.4f" % (name, R["calib"][name][1], R["calib"][name][2]))
P("Reference: Brier score of a constant prediction of the base rate = %.4f" % brier_score_loss(yt, np.full(len(yt), yt.mean())))

#@@ STEP 12_importance
step("12_importance")
pi = permutation_importance(best, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
sd = pd.Series(pi.importances_std, index=X_test.columns)
P("Permutation importance (drop in test PR-AUC when the feature is shuffled)")
for f, v in imp.items(): P("%-20s %.4f +/- %.4f" % (f, v, sd[f]))
R["imp"] = (imp, sd)

#@@ STEP 13_sanity
step("13_sanity")
shuf = cross_val_score(clone(best), X_train, y_train.sample(frac=1, random_state=SEED).values, cv=cv, scoring="average_precision")
P("Shuffled-label check: CV PR-AUC = %.3f (base rate = %.3f; real labels = %.3f)" % (shuf.mean(), y_train.mean(), search.best_score_))
seeds = []
for s_ in range(5):
    m = clone(best).set_params(clf__random_state=s_).fit(X_train, y_train)
    seeds.append(average_precision_score(y_test, m.predict_proba(X_test)[:, 1]))
P("Seed stability (5 seeds) test PR-AUC: " + ", ".join("%.3f" % v for v in seeds) + " -> mean %.3f, std %.4f" % (np.mean(seeds), np.std(seeds)))
r2 = np.random.default_rng(0); Xn = r2.normal(size=(100, 2000)); yn = r2.integers(0, 2, 100)
wrong = cross_val_score(LogisticRegression(max_iter=500), SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=5).mean()
right = cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=20)), ("lr", LogisticRegression(max_iter=500))]), Xn, yn, cv=5).mean()
P("Leakage demo (pure noise, 100 rows x 2000 features, true accuracy = 50%):")
P("  feature selection BEFORE cross-validation : accuracy = %.2f  (falsely high)" % wrong)
P("  feature selection INSIDE the pipeline     : accuracy = %.2f  (honest)" % right)
R["leak"] = (wrong, right)

#@@ STEP 14_persist
step("14_persist")
joblib.dump({"model": best, "threshold": float(f1_th)}, "churn_model_v1.joblib")
art = joblib.load("churn_model_v1.joblib")
same = np.allclose(art["model"].predict_proba(X_test)[:, 1], proba)
P("Saved churn_model_v1.joblib (%.0f KB); reloaded predictions identical: %s" % (os.path.getsize("churn_model_v1.joblib")/1024, same))
R["y_test"] = yt; R["f1_th"] = f1_th
pickle.dump(R, open("results.pkl", "wb"))
P("DONE")
