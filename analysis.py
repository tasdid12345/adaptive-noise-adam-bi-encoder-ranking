"""Fixed anticorrelated-noise Adam vs validation-tuned, adaptively scaled noise Adam (linear bi-encoder, 20NG)."""
import json, os, sys, time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

SEED, D, T, B, LR, TAU = 42, 128, 300, 128, 3e-3, 0.1
SIGMA_FIXED = 1e-3                      # hand-chosen noise std of the baseline
FINAL_SEEDS, TUNE_SEEDS = [42, 43, 44], [1, 2]
np.random.seed(SEED)

# ---------- data (download with retry/backoff) ----------
for i in range(4):
    try:
        raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        break
    except Exception as e:
        print("download failed:", e); time.sleep(2 ** i)
else:
    sys.exit("could not download 20 Newsgroups")
docs, labels = np.array(raw.data, dtype=object), raw.target
dev, test = train_test_split(np.arange(len(docs)), test_size=0.25, stratify=labels, random_state=SEED)
tr, va = train_test_split(dev, train_size=2000, test_size=1000, stratify=labels[dev], random_state=SEED)
test = np.random.default_rng(SEED).permutation(test)[:3000]   # fixed test pool (candidates)
vec = TfidfVectorizer(max_features=20000, sublinear_tf=True, stop_words="english", min_df=2).fit(docs[tr])
svd = TruncatedSVD(D, random_state=SEED).fit(vec.transform(docs[tr]))
sc = StandardScaler().fit(svd.transform(vec.transform(docs[tr])))
feat = lambda ix: sc.transform(svd.transform(vec.transform(docs[ix])))
Xtr, Xva, Xte = feat(tr), feat(va), feat(test)
ytr, yva, yte = labels[tr], labels[va], labels[test]
print(f"docs={len(docs)} train={len(tr)} val={len(va)} test pool={len(test)}")

# ---------- evaluation: queries are the first docs of a pool, candidates are the whole pool ----------
def evaluate(W, X, y, nq):
    E = X @ W; E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E[:nq] @ E.T; S[np.arange(nq), np.arange(nq)] = -np.inf   # drop the query itself
    order = np.argsort(-S, axis=1)
    rel = y[order] == y[:nq, None]
    nrel = (y[None, :] == y[:nq, None]).sum(1) - 1
    disc = 1 / np.log2(np.arange(2, 12))
    ndcg = (rel[:, :10] * disc).sum(1) / np.cumsum(disc)[np.minimum(nrel, 10) - 1]
    mrr = 1 / (rel.argmax(1) + 1)
    rec = rel[:, :10].sum(1) / nrel
    return dict(ndcg=ndcg.mean(), mrr=mrr.mean(), recall=rec.mean())

# ---------- model: linear bi-encoder, in-batch contrastive loss, Adam + anticorrelated noise ----------
cls_order = np.argsort(ytr, kind="stable"); cls_cnt = np.bincount(ytr); cls_start = np.cumsum(cls_cnt) - cls_cnt

def train(seed, mode, s=0.0):
    """mode: 'none' (plain Adam), 'fixed' (sigma=SIGMA_FIXED), 'adaptive' (sigma_t = s*lr*|g_t|/EMA|g|*decay_t)."""
    rng, nrng = np.random.default_rng(seed), np.random.default_rng(seed + 10_000)
    W = rng.normal(0, D ** -0.5, (D, D)); m = np.zeros_like(W); v = np.zeros_like(W)
    prev = np.zeros_like(W); gbar = None
    for t in range(1, T + 1):
        a = rng.integers(0, len(Xtr), B); ya = ytr[a]
        p = cls_order[cls_start[ya] + (rng.random(B) * cls_cnt[ya]).astype(int)]   # same-class positive
        xa, xp = Xtr[a], Xtr[p]
        u, w_ = xa @ W, xp @ W
        nu, nw = np.linalg.norm(u, axis=1, keepdims=True), np.linalg.norm(w_, axis=1, keepdims=True)
        q, d = u / nu, w_ / nw
        S = q @ d.T / TAU
        S[(ya[:, None] == ya[None, :]) & ~np.eye(B, dtype=bool)] = -np.inf   # mask false negatives
        P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        dS = (P - np.eye(B)) / B
        dq, dd = dS @ d / TAU, dS.T @ q / TAU
        du = (dq - q * (q * dq).sum(1, keepdims=True)) / nu
        dw = (dd - d * (d * dd).sum(1, keepdims=True)) / nw
        g = xa.T @ du + xp.T @ dw
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        W -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if mode == "none":
            continue
        gn = np.linalg.norm(g); gbar = gn if gbar is None else 0.9 * gbar + 0.1 * gn
        sigma = SIGMA_FIXED if mode == "fixed" else s * LR * (gn / gbar) * (1 - t / T)
        xi = nrng.standard_normal(W.shape)
        W += sigma * (xi - prev); prev = xi          # anticorrelated perturbation
    return W

# ---------- small GP Bayesian search of log10(s) on validation nDCG@10 ----------
def val_score(ls):
    return np.mean([evaluate(train(sd, "adaptive", 10 ** ls), Xva, yva, 500)["ndcg"] for sd in TUNE_SEEDS])

grid = np.linspace(-2, 1.5, 200)
xs = list(np.linspace(-2, 1.5, 4)); ys = [val_score(x) for x in xs]
print("init search:", [round(y, 4) for y in ys])
for it in range(6):
    gp = GaussianProcessRegressor(ConstantKernel() * Matern(length_scale=1.0, nu=2.5), alpha=1e-6,
                                  normalize_y=True, n_restarts_optimizer=2, random_state=SEED)
    gp.fit(np.array(xs)[:, None], ys)
    mu, sd = gp.predict(grid[:, None], return_std=True)
    z = (mu - max(ys)) / np.maximum(sd, 1e-9)
    ei = (mu - max(ys)) * norm.cdf(z) + sd * norm.pdf(z)
    ei[[np.any(np.isclose(g_, xs)) for g_ in grid]] = -1
    xs.append(float(grid[ei.argmax()])); ys.append(val_score(xs[-1]))
    print(f"BO step {it + 1}: log10 s={xs[-1]:.3f} val nDCG@10={ys[-1]:.4f}")
best_ls = xs[int(np.argmax(ys))]; s_best = 10 ** best_ls

# ---------- final: test set used once ----------
res = {}
for name, mode, s in [("baseline", "fixed", 0.0), ("improved", "adaptive", s_best), ("no_noise", "none", 0.0)]:
    runs = [evaluate(train(sd, mode, s), Xte, yte, 1000) for sd in FINAL_SEEDS]
    res[name] = {k: [r[k] for r in runs] for k in runs[0]}
    print(name, {k: round(float(np.mean(x)), 4) for k, x in res[name].items()})

mean = lambda n, k: float(np.mean(res[n][k])); std = lambda n, k: float(np.std(res[n][k]))
metrics = {"baseline_score": mean("baseline", "ndcg"), "improved_score": mean("improved", "ndcg"),
           "baseline_score_std": std("baseline", "ndcg"), "improved_score_std": std("improved", "ndcg"),
           "no_noise_score": mean("no_noise", "ndcg"), "no_noise_score_std": std("no_noise", "ndcg")}
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
for n in res:
    metrics[f"{n}_mrr"] = mean(n, "mrr"); metrics[f"{n}_recall_at_10"] = mean(n, "recall")
metrics.update(tuned_noise_scale_s=float(s_best), best_val_ndcg=float(max(ys)),
               fixed_sigma=SIGMA_FIXED, n_seeds=len(FINAL_SEEDS))
if not all(np.isfinite(list(metrics.values()))) or metrics["no_noise_score"] <= 0:
    sys.exit("non-finite or meaningless metrics")

out = {
    "question": "Can a validation-tuned, adaptively scaled noise level in Adam beat a hand-chosen fixed anticorrelated-noise Adam at training a small bi-encoder ranker, measured by nDCG@10 on held-out queries?",
    "method": "Linear bi-encoder on TF-IDF+SVD(128) features of 20 Newsgroups, in-batch contrastive loss, Adam with anticorrelated parameter noise; fixed vs GP-tuned adaptive noise scale.",
    "dataset": {"name": "20 Newsgroups (TF-IDF + TruncatedSVD 128-d)", "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html",
                "n_rows": int(len(docs)), "n_train": int(len(tr)), "n_test": int(len(test)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "nDCG@10", "higher_is_better": True,
    "baseline": "Adam with anticorrelated Gaussian parameter noise (sigma_t*(xi_t - xi_{t-1})) at a hand-chosen fixed sigma.",
    "improved": "Same model/loss/steps; noise scale chosen by a GP Bayesian search on validation nDCG@10 and scaled per step by gradient norm (relative to its EMA) and a linear decay.",
    "metrics": metrics,
    "notes": "Frozen-encoder stand-in (TF-IDF+SVD) and a linear model, not a transformer; relevance = same category. Seeds 42-44 (3 seeds), not 5. Validation set is disjoint from the 2,000 training docs and from the test pool; the noise-scale search used its own seeds (1,2) and only validation data. The baseline sigma is hand-chosen and untuned, so the improved model also receives a tuning budget the baseline does not; the training step budget is identical. Test queries are the first 1,000 docs of a 3,000-doc test pool, candidates are the pool. Differences may be within seed noise.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    names = ["no_noise", "baseline", "improved"]
    plt.bar(names, [mean(n, "ndcg") for n in names], yerr=[std(n, "ndcg") for n in names])
    plt.ylabel("test nDCG@10"); plt.tight_layout(); plt.savefig("figures/ndcg.png", dpi=120)
except Exception as e:
    print("plot skipped:", e)
print("done")
