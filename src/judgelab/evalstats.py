"""Offline evaluation methodology for a multi-label classifier compared against a baseline (numpy only).

The methods here were used to evaluate a fine-tuned multi-label route classifier against a baseline on out-of-fold (OOF)
predictions. The data, labels and models are not published; `synthetic_oof` generates a demo dataset with the same shape.

  auc / macro_auc        Mann-Whitney AUC (ties count one half), macro mean over routes; items with a missing label
                         (NaN) are skipped per route.
  paired_bootstrap       paired item bootstrap of macro AUC differences: the SAME resampled items are used for every arm
                         and every route, so arm-vs-arm noise from different resamples cancels.
  platt_fit / crossfit_platt
                         1-D logistic recalibration p = sigmoid(a*z + b), fitted by Newton steps; cross-fitted by fold
                         (fit on the other folds, predict the held-out fold), so the recalibrated scores stay out-of-sample.
  ece / ece_null         expected calibration error over equal-count bins, and its null distribution if the stated
                         probabilities were exactly right (labels drawn Bernoulli(p), same bins). Rule: RECALIBRATE when the
                         observed ECE exceeds the null 95th percentile.
  wilson                 Wilson score interval for a proportion.
  predictive_band        expected range of a difference measured once on a future dev set of size k drawn from the same
                         population: D_full + D(k drawn with replacement) - D(full-size bootstrap). Subsets drawn from the
                         n items already used for the point estimate overlap it, so their spread is too small by about
                         sqrt((1 + k/n) / (1 - k/n)); the predictive form avoids that.
"""
import math

import numpy as np

SAT_LOGODDS = 9.210  # |log-odds| at which p is within 1e-4 of 0 or 1


# ------------------------------------------------------------------------------------------ ranking
def rank_avg(x):
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(score, y):
    """Mann-Whitney AUC, ties count one half. y in {0, 1}; NaN labels are dropped. NaN when one class is absent."""
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    keep = ~np.isnan(y)
    score, y = score[keep], y[keep]
    p = y == 1
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_avg(score)
    return float((r[p].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro_auc(scores, labels):
    """scores, labels: dict route -> array over the same items. Returns (macro, {route: auc})."""
    per = {r: auc(scores[r], labels[r]) for r in labels}
    return float(np.mean(list(per.values()))), per


def _weighted_auc(score, y, w):
    """AUC with integer item multiplicities w (a bootstrap resample) without materialising the resample."""
    keep = (~np.isnan(y)) & (w > 0)
    s, yy, ww = score[keep], y[keep], w[keep].astype(float)
    pos = yy == 1
    n1, n0 = ww[pos].sum(), ww[~pos].sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s, pos, ww = s[order], pos[order], ww[order]
    # weighted average ranks: each distinct score value occupies a block of cumulative weight
    uniq, start = np.unique(s, return_index=True)
    ends = np.append(start[1:], len(s))
    ranks = np.empty(len(s))
    cum = 0.0
    for a, b in zip(start, ends):
        wsum = ww[a:b].sum()
        ranks[a:b] = cum + (wsum + 1.0) / 2.0
        cum += wsum
    return float(((ranks * ww)[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_bootstrap(arms, labels, n_boot=2000, seed=20261011):
    """arms: dict arm -> dict route -> scores; labels: dict route -> labels (NaN = not labelled).

    Returns dict arm -> array (n_boot,) of macro AUC on resampled items. Every arm and route uses the same multiplicities in
    every replicate, so `out[x] - out[y]` is the paired bootstrap distribution of the difference.
    """
    routes = list(labels)
    n = len(labels[routes[0]])
    rng = np.random.default_rng(seed)
    L = {r: np.asarray(labels[r], float) for r in routes}
    S = {a: {r: np.asarray(arms[a][r], float) for r in routes} for a in arms}
    out = {a: np.empty(n_boot) for a in arms}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        for a in arms:
            out[a][b] = float(np.mean([_weighted_auc(S[a][r], L[r], w) for r in routes]))
    return out


def summarize(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    return {"mean": float(arr.mean()), "lo95": float(np.percentile(arr, 2.5)), "hi95": float(np.percentile(arr, 97.5)),
            "n_valid": int(len(arr))}


# ------------------------------------------------------------------------------------------ calibration
def sigmoid(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logloss_from_z(z, y):
    """Mean binary log-loss computed stably from log-odds (no clipping)."""
    s = 2.0 * np.asarray(y, float) - 1.0
    return float(np.mean(np.logaddexp(0.0, -s * np.asarray(z, float))))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


class Reliability:
    """Equal-count bins over items sorted by score (stable sort; ties keep item order)."""

    def __init__(self, z, nbins=10):
        z = np.asarray(z, float)
        self.n, self.nbins = len(z), nbins
        self.order = np.argsort(z, kind="mergesort")
        base, extra = divmod(self.n, nbins)
        sizes = np.array([base + (1 if i < extra else 0) for i in range(nbins)])
        self.ends = np.cumsum(sizes)
        self.starts = self.ends - sizes
        self.size = sizes.astype(float)
        self.bin_of = np.empty(self.n, dtype=int)
        for b, (s, e) in enumerate(zip(self.starts, self.ends)):
            self.bin_of[self.order[s:e]] = b

    def means(self, v):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(self.bin_of, weights=np.asarray(v, float), minlength=self.nbins) / self.size

    def ece(self, p, y):
        m = self.size > 0
        return float(np.sum((self.size / self.n * np.abs(self.means(p) - self.means(y)))[m]))

    def table(self, p, y):
        mp, ro = self.means(p), self.means(y)
        return [{"bin": b + 1, "n": int(self.size[b]), "mean_pred": float(mp[b]), "obs_rate": float(ro[b]),
                 "gap": float(mp[b] - ro[b])} for b in range(self.nbins) if self.size[b]]


def ece(p, y, z=None, nbins=10):
    """ECE with equal-count bins ordered by z (default: by p)."""
    return Reliability(p if z is None else z, nbins).ece(p, y)


def ece_null(p, z=None, nbins=10, nsim=1000, seed=0):
    """ECE distribution if labels were Bernoulli(p) with the same binning: what a perfectly calibrated model would show."""
    p = np.asarray(p, float)
    rel = Reliability(p if z is None else z, nbins)
    rng = np.random.default_rng(seed)
    vals = np.array([rel.ece(p, (rng.random(len(p)) < p).astype(float)) for _ in range(nsim)])
    return {"mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99))}


def platt_fit(z, y, ridge=1e-6, iters=100):
    """Logistic regression y ~ sigmoid(a*z + b) by Newton steps with backtracking; tiny ridge on a. Returns (a, b)."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    s = 2.0 * y - 1.0
    a, b = 1.0, 0.0

    def nll(a_, b_):
        return float(np.sum(np.logaddexp(0.0, -s * (a_ * z + b_))) + 0.5 * ridge * a_ * a_)

    cur = nll(a, b)
    for _ in range(iters):
        p = sigmoid(a * z + b)
        r = p - y
        ga, gb = float(np.sum(r * z)) + ridge * a, float(np.sum(r))
        w = p * (1.0 - p)
        haa, hab, hbb = float(np.sum(w * z * z)) + ridge, float(np.sum(w * z)), float(np.sum(w)) + 1e-12
        det = haa * hbb - hab * hab
        if det <= 1e-18:
            break
        da, db = -(hbb * ga - hab * gb) / det, -(-hab * ga + haa * gb) / det
        t = 1.0
        while t > 1e-6:
            new = nll(a + t * da, b + t * db)
            if new <= cur + 1e-12:
                break
            t *= 0.5
        else:
            break
        a, b, cur = a + t * da, b + t * db, new
        if abs(t * da) + abs(t * db) < 1e-10:
            break
    return float(a), float(b)


def crossfit_platt(z, y, folds):
    """Out-of-fold recalibrated log-odds: for each fold, fit Platt on the other folds and apply it to this one."""
    z, y, folds = np.asarray(z, float), np.asarray(y, float), np.asarray(folds)
    zc = np.empty(len(z))
    fits = {}
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        a, b = (1.0, 0.0) if y[tr].min() == y[tr].max() else platt_fit(z[tr], y[tr])
        fits[f] = (a, b)
        zc[te] = a * z[te] + b
    return zc, fits


def wilson(k, n, zc=1.96):
    if n == 0:
        return None, None
    ph = k / n
    den = 1 + zc * zc / n
    c = (ph + zc * zc / (2 * n)) / den
    h = zc * math.sqrt(ph * (1 - ph) / n + zc * zc / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def calibration_report(z, y, folds, nbins=10, nsim=1000, seed=0):
    """Raw vs cross-fitted calibration of one route. verdict RECALIBRATE iff raw ECE > null p95."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    p = sigmoid(z)
    zc, fits = crossfit_platt(z, y, folds)
    pc = sigmoid(zc)
    raw_ece, raw_null = ece(p, y, z, nbins), ece_null(p, z, nbins, nsim, seed)
    cf_ece, cf_null = ece(pc, y, zc, nbins), ece_null(pc, zc, nbins, nsim, seed + 1)
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "brier_raw": brier(p, y), "brier_crossfit": brier(pc, y),
        "logloss_raw": logloss_from_z(z, y), "logloss_crossfit": logloss_from_z(zc, y),
        "ece_raw": raw_ece, "ece_raw_null_p95": raw_null["p95"],
        "ece_crossfit": cf_ece, "ece_crossfit_null_p95": cf_null["p95"],
        "saturated_share": float(np.mean(np.abs(z) >= SAT_LOGODDS)),
        "platt_full": platt_fit(z, y), "platt_by_fold": fits,
        "verdict": "RECALIBRATE" if raw_ece > raw_null["p95"] else "OK",
    }


# ------------------------------------------------------------------------------------------ predictive band
def predictive_band(arm, base, labels, k, n_sim=2000, seed=20261012):
    """Distribution of (arm - base) macro AUC expected from ONE future read on k items from the same population.

    Each replicate: D_full + [D on k items drawn with replacement] - [D on a full-size bootstrap]. Returns summarize()
    of the replicates plus the share above 0. Limits: item sampling only (no dataset shift, no model-refit variance).
    """
    routes = list(labels)
    n = len(labels[routes[0]])
    L = {r: np.asarray(labels[r], float) for r in routes}
    A = {r: np.asarray(arm[r], float) for r in routes}
    B = {r: np.asarray(base[r], float) for r in routes}
    d_full = macro_auc(A, L)[0] - macro_auc(B, L)[0]
    rng = np.random.default_rng(seed)

    def d(w):
        return float(np.mean([_weighted_auc(A[r], L[r], w) - _weighted_auc(B[r], L[r], w) for r in routes]))

    vals = []
    for _ in range(n_sim):
        wk = np.bincount(rng.integers(0, n, k), minlength=n)
        wn = np.bincount(rng.integers(0, n, n), minlength=n)
        v = d_full + d(wk) - d(wn)
        if not math.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    out = summarize(vals)
    out.update(point_full=d_full, share_above_0=float(np.mean(vals > 0)), k=k, n=n)
    return out


# ------------------------------------------------------------------------------------------ synthetic demo data
ROUTES = ("route_a", "route_b", "route_c", "route_d", "route_e")


def synthetic_oof(n=600, folds=5, seed=7, overconfidence=3.0, missing=0.05):
    """Synthetic OOF predictions with the shape of the real study: 5 routes, binary labels (some missing), a baseline arm
    and a stronger arm whose log-odds are over-confident by `overconfidence` (true slope 1/overconfidence)."""
    rng = np.random.default_rng(seed)
    labels, base, arm = {}, {}, {}
    for i, r in enumerate(ROUTES):
        rate = 0.12 + 0.09 * i
        latent = rng.normal(0, 1, n)
        y = (latent + rng.normal(0, 0.9, n) > np.quantile(latent, 1 - rate)).astype(float)
        sig_b = 1.2 + 0.1 * i
        sig_a = sig_b + 0.6
        zb = sig_b * (y - 0.5) * 2 + rng.normal(0, 1.6, n)
        za = sig_a * (y - 0.5) * 2 + rng.normal(0, 1.4, n)
        y[rng.random(n) < missing] = np.nan
        labels[r], base[r], arm[r] = y, zb, za * overconfidence
    fold = np.arange(n) % folds
    rng.shuffle(fold)
    return {"labels": labels, "base": base, "arm": arm, "folds": fold + 1}
