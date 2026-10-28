import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
R = pickle.load(open("results.pkl", "rb")); y = R["y_test"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
B, O, G, Rd = "#1F4E79", "#E07B39", "#4C9F70", "#C00000"

# Fig A: PR curves + threshold/cost
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
cols = {"Logistic regression": B, "Random forest": G, "Gradient boosting (tuned)": O}
for n_, c in cols.items():
    p, r, _ = precision_recall_curve(y, R["test_proba"][n_])
    ax[0].plot(r, p, color=c, lw=1.8, label=f"{n_} (AP={R['test_scores'][n_][0]:.2f})")
ax[0].axhline(y.mean(), color="#999999", ls="--", lw=1.2, label="Baseline = churn rate (0.20)")
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("Precision-recall curves (test set)")
ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_ylim(0, 1.02)
oof = R["oof"]; ytr = None
import pandas as pd
ths = np.linspace(0.02, 0.95, 94)
# recompute cost curve on test probabilities
pt = R["test_proba"]["Gradient boosting (tuned)"]
cost = []
for t in ths:
    pr = pt >= t; fn = ((~pr) & (y == 1)).sum(); fp = (pr & (y == 0)).sum()
    cost.append((300*fn + 40*fp) / len(y) * 1000)
ax[1].plot(ths, np.array(cost)/1000, color=B, lw=1.8)
ax[1].axhline(300*y.mean(), color="#999999", ls="--", lw=1.2); ax[1].text(0.55, 300*y.mean()+1, "No model: $61.0k", fontsize=7, color="#555")
XY = {"0.50": (6, 6), "F1-opt": (8, 6), "cost-opt": (10, -12)}
for name, t, c in [("0.50", 0.50, Rd), ("F1-opt %.2f" % R["thr"]["f1"], R["thr"]["f1"], G), ("cost-opt %.2f" % R["thr"]["cost"], R["thr"]["cost"], O)]:
    i = np.argmin(abs(ths - t)); ax[1].scatter([ths[i]], [cost[i]/1000], color=c, zorder=5, s=28)
    ax[1].annotate(name, (ths[i], cost[i]/1000), textcoords="offset points", xytext=XY[name.split()[0]], fontsize=7, color=c)
ax[1].set_xlabel("Decision threshold"); ax[1].set_ylabel("Expected cost per 1,000 customers ($ thousands)")
ax[1].set_ylim(15, 64); ax[1].set_title("Choosing the threshold by business cost")
plt.tight_layout(); plt.savefig("figA.png", facecolor="white"); plt.close()

# Fig B: validation curve + learning curve
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
l, tr, va = R["valcurve"]
ax[0].plot(l, tr.mean(1), "o-", color=O, label="Training score"); ax[0].plot(l, va.mean(1), "o-", color=B, label="Cross-validation score")
ax[0].fill_between(l, va.mean(1)-va.std(1), va.mean(1)+va.std(1), color=B, alpha=.15)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(l); ax[0].set_xticklabels(l)
ax[0].set_xlabel("max_leaf_nodes (model complexity)"); ax[0].set_ylabel("PR-AUC"); ax[0].set_title("Validation curve: overfitting as complexity grows"); ax[0].legend(fontsize=8)
s, tr, va = R["learning"]
ax[1].plot(s, tr.mean(1), "o-", color=O, label="Training score"); ax[1].plot(s, va.mean(1), "o-", color=B, label="Cross-validation score")
ax[1].fill_between(s, va.mean(1)-va.std(1), va.mean(1)+va.std(1), color=B, alpha=.15)
ax[1].set_xlabel("Number of training rows"); ax[1].set_ylabel("PR-AUC"); ax[1].set_title("Learning curve (default gradient boosting)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figB.png", facecolor="white"); plt.close()

# Fig C: confusion matrix + permutation importance
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200, gridspec_kw={"width_ratios": [1, 1.3]})
tn, fp, fn, tp = R["cm"]; cm = np.array([[tn, fp], [fn, tp]])
ax[0].imshow(cm, cmap="Blues"); ax[0].grid(False)
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=14, color="white" if cm[i, j] > 700 else "black", fontweight="bold")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Pred: stay", "Pred: churn"]); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["Actual: stay", "Actual: churn"])
ax[0].set_title("Confusion matrix (test, threshold %.2f)" % R["f1_th"])
imp, sd = R["imp"]; imp = imp.head(8)[::-1]
ax[1].barh(imp.index, imp.values, xerr=sd[imp.index].values, color=B, alpha=.85, capsize=2)
ax[1].set_xlabel("Drop in test PR-AUC when shuffled"); ax[1].set_title("Permutation feature importance")
plt.tight_layout(); plt.savefig("figC.png", facecolor="white"); plt.close()

# Fig D: calibration
LBL = {"Tuned (no class weights)": "Tuned model", "Tuned + class_weight='balanced'": "With balanced class weights", "Tuned + isotonic calibration": "With isotonic calibration"}
fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.8), dpi=200)
ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
for (n_, (p_, br, e)), c in zip(R["calib"].items(), [B, Rd, G]):
    fr, mp = calibration_curve(y, p_, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, "o-", color=c, lw=1.5, ms=4, label=f"{LBL[n_]} (ECE {e:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed churn rate"); ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=6.5, loc="upper left")
plt.tight_layout(); plt.savefig("figD.png", facecolor="white"); plt.close()
print("ok")
