"""
Direct RUL regression with a parametric predictive head.

Instead of modelling the degradation path and taking a first-passage time, the
remaining useful life is regressed directly and the model outputs the PARAMETERS
of a distribution over RUL, trained by maximum likelihood. The Birnbaum-Saunders
head is the one indicated by theory: the first-passage time of cumulative fatigue
damage converges to a BS law, so BS is the shape the RUL predictive distribution
should have.

Every head sits on an identical backbone and is trained identically, so
differences between heads are attributable to the distribution family alone.

Heads: BS, BS-t, Weibull, lognormal, inverse Gaussian, Gaussian.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

LOG2PI = math.log(2.0 * math.pi)
EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════
# 1.  Predictive heads
# ══════════════════════════════════════════════════════════════════════════
def _xi(u):
    s = torch.sqrt(u)
    return s - 1.0 / s


def _log_xi_prime(u):
    # xi'(u) = (u^{-1/2} + u^{-3/2}) / 2
    return torch.log(0.5 * (u ** -0.5 + u ** -1.5))


class Head:
    """A two-parameter distribution over RUL > 0.

    `raw` is the (N,2) network output; each head maps it to its own natural
    parameters, and provides log-density, mean, variance, quantiles and samples.
    """
    name = "abstract"
    n_out = 2

    def params(self, raw):
        raise NotImplementedError

    def logpdf(self, t, p):
        raise NotImplementedError

    def mean(self, p):
        raise NotImplementedError

    def var(self, p):
        raise NotImplementedError

    def ppf(self, q, p):
        raise NotImplementedError

    def sample(self, p, n, gen=None):
        u = torch.rand(p[0].shape[0], n, generator=gen, device=p[0].device)
        return self.ppf(u, tuple(x[:, None] for x in p))


class BSHead(Head):
    """RUL ~ BS(alpha, beta).  log alpha, log beta = network outputs.

    This generalises the log-linear BS regression of Rieck and Nedelman (1991)
    to a non-linear feature map, and lets the shape parameter be
    covariate-dependent as well as the scale.
    """
    name = "BS"

    def params(self, raw):
        a = torch.exp(torch.clamp(raw[:, 0], -3.0, 1.5))
        b = torch.exp(torch.clamp(raw[:, 1], -7.0, 3.0))
        return a, b

    def logpdf(self, t, p):
        a, b = p
        u = torch.clamp(t / b, EPS, 1e8)
        z = _xi(u) / a
        return _log_xi_prime(u) - torch.log(a * b) - 0.5 * z ** 2 - 0.5 * LOG2PI

    def mean(self, p):
        a, b = p
        return b * (1.0 + a ** 2 / 2.0)

    def var(self, p):
        a, b = p
        return b ** 2 * a ** 2 * (1.0 + 1.25 * a ** 2)

    def ppf(self, q, p):
        a, b = p
        z = torch.erfinv(2.0 * torch.clamp(q, 1e-6, 1 - 1e-6) - 1.0) * math.sqrt(2.0)
        return b / 4.0 * (a * z + torch.sqrt(4.0 + (a * z) ** 2)) ** 2


class BSTHead(BSHead):
    """BS-Student-t head: the generalised BS family with g = t_nu, nu fixed.

    Heavier tails than BS at the same (alpha, beta); included because bearing
    lifetimes are notoriously dispersed and the extra tail mass costs one
    hyper-parameter rather than one more learned output.
    """
    name = "BS-t"

    def __init__(self, nu: float = 6.0):
        # nu > 4 is required: the scale mixture T | U ~ BS(alpha/sqrt(U), beta)
        # with U ~ Gamma(nu/2, nu/2) has E[1/U^2] = nu^2/((nu-2)(nu-4)), so the
        # predictive variance -- and hence the aleatory component of the
        # uncertainty decomposition -- is infinite for nu <= 4.
        if nu <= 4.0:
            raise ValueError("BS-t head needs nu > 4 for a finite variance")
        self.nu = nu
        self._lc = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                    - 0.5 * math.log(nu * math.pi))
        self._c1 = nu / (nu - 2.0)                       # E[1/U]
        self._c2 = nu ** 2 / ((nu - 2.0) * (nu - 4.0))   # E[1/U^2]

    def logpdf(self, t, p):
        a, b = p
        u = torch.clamp(t / b, EPS, 1e8)
        z = _xi(u) / a
        return (_log_xi_prime(u) - torch.log(a * b) + self._lc
                - (self.nu + 1) / 2 * torch.log1p(z ** 2 / self.nu))

    def mean(self, p):
        # E[T] = E_U[ beta (1 + (alpha^2/U)/2) ] = beta (1 + alpha^2 c1 / 2)
        a, b = p
        return b * (1.0 + self._c1 * a ** 2 / 2.0)

    def var(self, p):
        """Law of total variance over the mixing variable U.

        Var(T|U) = beta^2 [ alpha^2/U + (5/4) alpha^4/U^2 ]  and
        E[T|U]   = beta [ 1 + alpha^2/(2U) ],  so with c1 = E[1/U] and
        c2 = E[1/U^2],

            Var(T) = beta^2 [ alpha^2 c1 + (5/4) alpha^4 c2
                              + (alpha^4/4)(c2 - c1^2) ].

        Substituting the BS-t effective shape alpha*sqrt(c1) into the plain BS
        variance formula, as one might be tempted to do, is wrong: it misses the
        contribution of the spread of 1/U.
        """
        a, b = p
        a2 = a ** 2
        a4 = a2 ** 2
        return b ** 2 * (a2 * self._c1 + 1.25 * a4 * self._c2
                         + 0.25 * a4 * (self._c2 - self._c1 ** 2))

    def ppf(self, q, p):
        a, b = p
        qq = torch.clamp(q, 1e-6, 1 - 1e-6)
        # Student-t quantile via the Cornish-Fisher-free route: use the
        # scale-mixture, t_nu = Z / sqrt(V/nu); invert numerically by bisection
        z = _t_ppf(qq, self.nu)
        az = a * z
        return b / 4.0 * (az + torch.sqrt(4.0 + az ** 2)) ** 2


def _t_cdf(x, nu):
    """Student-t CDF. Used only for quantiles and sampling, never inside the
    training graph, so SciPy's implementation is used directly rather than a
    hand-rolled incomplete beta."""
    from scipy import stats as _st
    return torch.as_tensor(_st.t.cdf(x.detach().cpu().numpy(), nu),
                           dtype=x.dtype, device=x.device)


def _t_ppf(q, nu):
    """Student-t quantile; see _t_cdf on why SciPy is acceptable here."""
    from scipy import stats as _st
    return torch.as_tensor(_st.t.ppf(q.detach().cpu().numpy(), nu),
                           dtype=q.dtype, device=q.device)


class WeibullHead(Head):
    """RUL ~ Weibull(shape k, scale lam). The standard parametric head in
    time-to-event prognostics."""
    name = "Weibull"

    def params(self, raw):
        k = torch.exp(torch.clamp(raw[:, 0], -1.5, 2.5))
        lam = torch.exp(torch.clamp(raw[:, 1], -7.0, 3.0))
        return k, lam

    def logpdf(self, t, p):
        k, lam = p
        tt = torch.clamp(t, EPS, None)
        r = tt / lam
        return torch.log(k) - torch.log(lam) + (k - 1) * torch.log(r) - r ** k

    def mean(self, p):
        k, lam = p
        return lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))

    def var(self, p):
        k, lam = p
        g1 = torch.exp(torch.lgamma(1.0 + 1.0 / k))
        g2 = torch.exp(torch.lgamma(1.0 + 2.0 / k))
        return lam ** 2 * (g2 - g1 ** 2)

    def ppf(self, q, p):
        k, lam = p
        qq = torch.clamp(q, 1e-6, 1 - 1e-6)
        return lam * (-torch.log1p(-qq)) ** (1.0 / k)


class LogNormalHead(Head):
    name = "LogNormal"

    def params(self, raw):
        mu = torch.clamp(raw[:, 0], -7.0, 3.0)
        sig = torch.exp(torch.clamp(raw[:, 1], -3.0, 0.7))
        return mu, sig

    def logpdf(self, t, p):
        mu, sig = p
        tt = torch.clamp(t, EPS, None)
        z = (torch.log(tt) - mu) / sig
        return -torch.log(tt) - torch.log(sig) - 0.5 * LOG2PI - 0.5 * z ** 2

    def mean(self, p):
        mu, sig = p
        return torch.exp(mu + sig ** 2 / 2)

    def var(self, p):
        mu, sig = p
        return (torch.exp(sig ** 2) - 1) * torch.exp(2 * mu + sig ** 2)

    def ppf(self, q, p):
        mu, sig = p
        z = torch.erfinv(2.0 * torch.clamp(q, 1e-6, 1 - 1e-6) - 1.0) * math.sqrt(2.0)
        return torch.exp(mu + sig * z)


class IGHead(Head):
    name = "IGauss"

    def params(self, raw):
        mu = torch.exp(torch.clamp(raw[:, 0], -7.0, 3.0))
        lam = torch.exp(torch.clamp(raw[:, 1], -5.0, 7.0))
        return mu, lam

    def logpdf(self, t, p):
        mu, lam = p
        tt = torch.clamp(t, EPS, None)
        return (0.5 * torch.log(lam) - 0.5 * LOG2PI - 1.5 * torch.log(tt)
                - lam * (tt - mu) ** 2 / (2 * mu ** 2 * tt))

    def mean(self, p):
        return p[0]

    def var(self, p):
        mu, lam = p
        return mu ** 3 / lam

    def ppf(self, q, p, iters=60):
        mu, lam = p
        lo = torch.full_like(q, 1e-8)
        hi = 60.0 * mu + torch.ones_like(q)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            c = _ig_cdf(mid, mu, lam)
            lo = torch.where(c < q, mid, lo)
            hi = torch.where(c < q, hi, mid)
        return 0.5 * (lo + hi)


def _ndtr(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _ig_cdf(t, mu, lam):
    tt = torch.clamp(t, 1e-12, None)
    r = torch.sqrt(lam / tt)
    a = r * (tt / mu - 1.0)
    b = -r * (tt / mu + 1.0)
    return _ndtr(a) + torch.exp(torch.clamp(2 * lam / mu, None, 60.0)) * _ndtr(b)


class GaussianHead(Head):
    """Included as the naive baseline. It places mass on RUL < 0, which is
    reported rather than hidden."""
    name = "Gaussian"

    def params(self, raw):
        mu = torch.clamp(raw[:, 0], -10.0, 10.0)
        sig = torch.exp(torch.clamp(raw[:, 1], -3.0, 1.5))
        return mu, sig

    def logpdf(self, t, p):
        mu, sig = p
        z = (t - mu) / sig
        return -torch.log(sig) - 0.5 * LOG2PI - 0.5 * z ** 2

    def mean(self, p):
        return p[0]

    def var(self, p):
        return p[1] ** 2

    def ppf(self, q, p):
        mu, sig = p
        z = torch.erfinv(2.0 * torch.clamp(q, 1e-6, 1 - 1e-6) - 1.0) * math.sqrt(2.0)
        return mu + sig * z


HEADS = {"BS": BSHead, "BS-t": BSTHead, "Weibull": WeibullHead,
         "LogNormal": LogNormalHead, "IGauss": IGHead, "Gaussian": GaussianHead}


# ── Heads whose support matches a target in (0,1] ─────────────────────────
#  The six heads above have support (0,inf) or the whole line, wider than the
#  remaining fraction of life. The heads below respect the bound in training:
#  two bounded families, and the two first-passage laws truncated at one.
def _numeric_moments(head, p, k=200):
    """Mean and variance from k midpoint quantiles, for laws without closed forms."""
    u = ((torch.arange(k, dtype=p[0].dtype, device=p[0].device) + 0.5) / k)[None, :]
    x = head.ppf(u.expand(p[0].shape[0], k), tuple(t[:, None] for t in p))
    m = x.mean(1)
    return m, (x ** 2).mean(1) - m ** 2


class LogitNormalHead(Head):
    """logit(R) ~ N(mu, sigma^2): a bounded family on (0,1)."""
    name = "LogitNormal"

    def params(self, raw):
        mu = torch.clamp(raw[:, 0], -10.0, 10.0)
        sig = torch.exp(torch.clamp(raw[:, 1], -3.0, 1.5))
        return mu, sig

    def logpdf(self, t, p):
        mu, sig = p
        tt = torch.clamp(t, EPS, 1 - EPS)
        z = (torch.log(tt) - torch.log1p(-tt) - mu) / sig
        return -torch.log(tt) - torch.log1p(-tt) - torch.log(sig) - 0.5 * LOG2PI - 0.5 * z ** 2

    def ppf(self, q, p):
        mu, sig = p
        z = torch.erfinv(2.0 * torch.clamp(q, 1e-6, 1 - 1e-6) - 1.0) * math.sqrt(2.0)
        return torch.sigmoid(mu + sig * z)

    def mean(self, p):
        return _numeric_moments(self, p)[0]

    def var(self, p):
        return _numeric_moments(self, p)[1]


class BetaHead(Head):
    """R ~ Beta(a, b): the standard regression family for a bounded target."""
    name = "Beta"

    def params(self, raw):
        a = torch.exp(torch.clamp(raw[:, 0], -3.0, 5.0))
        b = torch.exp(torch.clamp(raw[:, 1], -3.0, 5.0))
        return a, b

    def logpdf(self, t, p):
        a, b = p
        tt = torch.clamp(t, EPS, 1 - EPS)
        return ((a - 1) * torch.log(tt) + (b - 1) * torch.log1p(-tt)
                + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b))

    def mean(self, p):
        a, b = p
        return a / (a + b)

    def var(self, p):
        a, b = p
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def ppf(self, q, p):
        # quantiles and sampling only, never in the training graph (cf. _t_ppf)
        from scipy import stats as _st
        a, b = p
        f = lambda x: x.detach().cpu().numpy()
        return torch.as_tensor(_st.beta.ppf(np.clip(f(q), 1e-6, 1 - 1e-6), f(a), f(b)),
                               dtype=q.dtype, device=q.device)


def _bs_cdf(t, a, b):
    return _ndtr(_xi(torch.clamp(t / b, EPS, 1e8)) / a)


def _ig_cdf_stable(t, mu, lam):
    """IG CDF with the second term in log space: exp(2 lam/mu) overflows for
    the concentrated laws a truncated head can learn."""
    tt = torch.clamp(t, 1e-12, None)
    r = torch.sqrt(lam / tt)
    a = r * (tt / mu - 1.0)
    b = -r * (tt / mu + 1.0)
    return _ndtr(a) + torch.exp(torch.clamp(2 * lam / mu + torch.special.log_ndtr(b), None, 0.0))


class TruncatedHead(Head):
    """A positive-support head conditioned on R <= 1 and fitted by the
    truncated likelihood  log f(y) - log F(1)."""

    def __init__(self, base, cdf):
        self.base, self.cdf = base, cdf
        self.name = base.name + "-trunc"

    def params(self, raw):
        return self.base.params(raw)

    def _logF1(self, p):
        one = torch.ones_like(p[0])
        return torch.log(torch.clamp(self.cdf(one, *p), 1e-12, 1.0))

    def logpdf(self, t, p):
        return self.base.logpdf(t, p) - self._logF1(p)

    def ppf(self, q, p):
        F1 = torch.exp(self._logF1(p))
        return torch.clamp(self.base.ppf(q * F1, p), None, 1.0)

    def mean(self, p):
        return _numeric_moments(self, p)[0]

    def var(self, p):
        return _numeric_moments(self, p)[1]


class _IGStable(IGHead):
    def ppf(self, q, p, iters=60):
        mu, lam = p
        lo = torch.full_like(q, 1e-8)
        hi = 60.0 * mu + torch.ones_like(q)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            c = _ig_cdf_stable(mid, mu, lam)
            lo = torch.where(c < q, mid, lo)
            hi = torch.where(c < q, hi, mid)
        return 0.5 * (lo + hi)


BOUNDED_HEADS = {
    "LogitNormal": LogitNormalHead,
    "Beta": BetaHead,
    "BS-trunc": lambda: TruncatedHead(BSHead(), _bs_cdf),
    "IGauss-trunc": lambda: TruncatedHead(_IGStable(), _ig_cdf_stable),
}


# ══════════════════════════════════════════════════════════════════════════
# 2.  Backbone
# ══════════════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    """Small MLP shared by every head. Deliberately small: the effective sample
    size is the number of run-to-failure records, not the number of time points."""

    def __init__(self, d_in, hidden=(48, 24), p_drop=0.10):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():                       # start near the data scale
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainCfg:
    #  hidden is part of the training configuration so that the
    #  representation-independence check can vary the backbone without forking
    #  the code the main results were produced with.
    hidden: tuple = (48, 24)
    epochs: int = 300
    lr: float = 3e-3
    weight_decay: float = 1e-3
    batch: int = 256
    patience: int = 40
    seed: int = 0


def train_head(head, Xtr, ytr, wtr, Xva, yva, cfg: TrainCfg):
    """Maximum-likelihood fit of one head. `w` weights each sample so that every
    bearing contributes equally regardless of record length."""
    torch.manual_seed(cfg.seed)
    model = MLP(Xtr.shape[1], hidden=cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    wtr_t = torch.tensor(wtr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    n = len(ytr)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(cfg.seed)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            p = head.params(model(Xtr_t[idx]))
            nll = -(head.logpdf(ytr_t[idx], p) * wtr_t[idx]).sum() / wtr_t[idx].sum()
            if not torch.isfinite(nll):
                continue
            opt.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            v = -head.logpdf(yva_t, head.params(model(Xva_t))).mean().item()
        if np.isfinite(v) and v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def predict(head, models, X, n_samp=400, seed=0):
    """Deep-ensemble prediction.

    The ensemble gives exactly the law-of-total-variance split used throughout
    this paper: the aleatory part is the mean of the per-member predictive
    variances, the epistemic part the variance of the per-member predictive
    means.
    """
    Xt = torch.tensor(X, dtype=torch.float32)
    means, varis, samples = [], [], []
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in models:
            p = head.params(m(Xt))
            means.append(head.mean(p))
            varis.append(head.var(p))
            samples.append(head.sample(p, n_samp // len(models), gen=g))
    M = torch.stack(means)                                   # (M,N)
    V = torch.stack(varis)
    mean = M.mean(0)
    v_al = V.mean(0)
    v_ep = M.var(0, unbiased=False)
    S = torch.cat(samples, dim=1)                            # (N, n_samp)
    S, _ = torch.sort(S, dim=1)
    return {"mean": mean.numpy(), "v_aleatory": v_al.numpy(),
            "v_epistemic": v_ep.numpy(), "samples": S.numpy(),
            "members_mean": M.numpy()}


# ══════════════════════════════════════════════════════════════════════════
# 3.  Proper scoring rules and calibration
# ══════════════════════════════════════════════════════════════════════════
def crps_from_samples(samples, y):
    """CRPS estimated from sorted predictive samples.

    CRPS = E|X - y| - 0.5 E|X - X'|, the second term computed in O(n log n) from
    the sorted sample via  E|X-X'| = (2/n^2) sum_i (2i - n + 1) x_(i).
    """
    n = samples.shape[1]
    t1 = np.abs(samples - y[:, None]).mean(axis=1)
    i = np.arange(1, n + 1)
    t2 = (2.0 / (n * n)) * ((2 * i - n - 1) * samples).sum(axis=1)
    return t1 - 0.5 * t2


def pit_values(samples, y):
    """Probability integral transform: the predictive CDF evaluated at the truth."""
    return (samples < y[:, None]).mean(axis=1)


def ks_uniform(p):
    """Kolmogorov-Smirnov distance of the PIT values from uniform. 0 is perfect."""
    p = np.sort(np.clip(p, 0, 1))
    n = len(p)
    e = (np.arange(1, n + 1)) / n
    return float(max(np.max(np.abs(p - e)), np.max(np.abs(p - (e - 1.0 / n)))))
