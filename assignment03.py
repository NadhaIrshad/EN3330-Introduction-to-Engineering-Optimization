"""
EN3330 Assignment 03 - Lagrangian duality for  min x^2  s.t.  x >= 1

    h(x) = 1 - x <= 0
    L(mu, x) = x^2 + mu (1 - x),  mu >= 0

Run:  python assignment03.py
All figures are saved to ./figures (PNG, 300 dpi) and numerical checks are
printed to the console.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})


# ----------------------------------------------------------------------------
# Problem definitions
# ----------------------------------------------------------------------------
def f(x):
    return x**2


def h(x):
    return 1 - x


def L(mu, x):
    return f(x) + mu * h(x)


def x_of_mu(mu):
    # dL/dx = 2x - mu = 0  ->  x(mu) = mu / 2
    return mu / 2


def d(mu):
    # d(mu) = L(mu, mu/2) = mu - mu^2 / 4
    return mu - mu**2 / 4


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  saved {path}")


# ----------------------------------------------------------------------------
# Q1(a)  L(mu, x) vs x for mu = 0,1,2,3 and the minimizers x(mu)
# ----------------------------------------------------------------------------
def q1a():
    print("\nQ1(a)  minimizers x(mu) = argmin_x L(mu, x) = mu/2")
    x = np.linspace(-2, 3.5, 500)
    mus = [0, 1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(mus)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for mu, c in zip(mus, colors):
        ax.plot(x, L(mu, x), color=c, lw=2, label=rf"$\mu={mu}$")
        xm = x_of_mu(mu)
        ax.plot(xm, L(mu, xm), "o", color=c, ms=8, mec="k", zorder=5)
        ax.annotate(rf"$x({mu})={xm:g}$", (xm, L(mu, xm)),
                    textcoords="offset points", xytext=(-10, -22), fontsize=9)
        print(f"  mu={mu}:  x(mu)={xm:.2f},  L(mu, x(mu)) = d(mu) = {L(mu, xm):.4f}")

    # trajectory of the minimizers: (mu/2, mu - mu^2/4)  ->  y = 2x - x^2
    mu_path = np.linspace(0, 3, 100)
    ax.plot(x_of_mu(mu_path), d(mu_path), "k--", lw=1.2,
            label=r"path of $(x(\mu), d(\mu))$")
    ax.axvline(1, color="r", ls=":", lw=1.5, label=r"constraint boundary $x=1$")
    ax.axvspan(-2, 1, color="r", alpha=0.06)

    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-0.5, 6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L(\mu, x)$")
    ax.set_title(r"Q1(a): $L(\mu,x)=x^2+\mu(1-x)$ for $\mu=0,1,2,3$")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "q1a_lagrangian_vs_x.png")


# ----------------------------------------------------------------------------
# Q1(b)  Dual function d(mu) = mu - mu^2/4
# ----------------------------------------------------------------------------
def q1b():
    mu = np.linspace(0, 5, 500)

    # numerical check: brute-force min over a fine x grid
    xg = np.linspace(-10, 10, 200001)
    d_num = np.array([L(m, xg).min() for m in mu])
    print(f"\nQ1(b)  max |d_analytic - d_numeric| = {np.max(np.abs(d(mu) - d_num)):.2e}")

    mu_star = mu[np.argmax(d(mu))]
    d_star = d(mu).max()
    print(f"  mu* = {mu_star:.4f}  (analytic: d'(mu) = 1 - mu/2 = 0 -> mu* = 2)")
    print(f"  d*  = {d_star:.4f}  (analytic: d(2) = 2 - 1 = 1)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mu, d(mu), "b", lw=2, label=r"$d(\mu)=\mu-\mu^2/4$")
    ax.plot(mu[::25], d_num[::25], "kx", ms=6, label="numerical $\\min_x L$")
    ax.plot(2, 1, "ro", ms=9, zorder=5)
    ax.annotate(r"$\mu^*=2,\ d^*=1$", (2, 1), textcoords="offset points",
                xytext=(10, 8), color="r")
    ax.axhline(1, color="gray", ls="--", lw=1, label=r"$p^*=f(x^*)=1$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$d(\mu)$")
    ax.set_title("Q1(b): Dual function")
    ax.legend()
    save(fig, "q1b_dual_function.png")


# ----------------------------------------------------------------------------
# Q1(c)  Primal function g(x) = max_{mu>=0} L(mu, x)
# ----------------------------------------------------------------------------
def g(x):
    x = np.asarray(x, dtype=float)
    # L is linear in mu with slope h(x): bounded above only if h(x) <= 0
    return np.where(h(x) <= 0, f(x), np.inf)


def q1c():
    x = np.linspace(-1, 3, 801)
    print("\nQ1(c)  g(x) = x^2 if x >= 1, +inf if x < 1")
    for xi in [-1, 0, 0.5, 0.99, 1, 2, 3]:
        print(f"  x={xi:5.2f}: g(x) = {float(g(xi))}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: L(mu, x) for growing mu -> upper envelope blows up for x < 1
    ax = axes[0]
    mus = [0, 1, 2, 5, 10, 20, 50]
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(mus)))
    for mu, c in zip(mus, colors):
        ax.plot(x, L(mu, x), color=c, lw=1.5, label=rf"$\mu={mu}$")
    ax.axvline(1, color="k", ls=":", lw=1.5)
    ax.axvspan(-1, 1, color="r", alpha=0.07)
    ax.text(-0.9, 85, "infeasible: $h(x)>0$\n$L\\to+\\infty$ as $\\mu\\to\\infty$",
            fontsize=9, color="darkred")
    ax.text(1.6, 85, "feasible: $h(x)\\leq0$\nmax at $\\mu=0$", fontsize=9)
    ax.set_ylim(-5, 110)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L(\mu,x)$")
    ax.set_title(r"$L(\mu,x)$ for increasing $\mu$")
    ax.legend(fontsize=8, ncol=2, loc="center right")

    # Right: g(x) itself
    ax = axes[1]
    xf = x[x >= 1]
    ax.plot(xf, f(xf), "b", lw=2.5, label=r"$g(x)=x^2$  ($x\geq1$)")
    ax.plot(x, f(x), "b:", lw=1, alpha=0.5, label=r"$f(x)=x^2$")
    ymax = 12
    ax.fill_between(x[x < 1], ymax, ymax + 1, color="r", alpha=0.4)
    for xa in np.linspace(-0.9, 0.9, 7):
        ax.annotate("", xy=(xa, ymax + 0.9), xytext=(xa, ymax - 1.5),
                    arrowprops=dict(arrowstyle="->", color="r"))
    ax.text(-0.75, ymax - 3, r"$g(x)=+\infty$ for $x<1$", color="r")
    ax.plot(1, 1, "go", ms=9, zorder=5)
    ax.annotate(r"$\min\ g(x)=g(1)=1$", (1, 1), textcoords="offset points",
                xytext=(10, -5), color="g")
    ax.axvline(1, color="k", ls=":", lw=1.5)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-0.5, ymax + 1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$g(x)$")
    ax.set_title(r"Primal function $g(x)=\max_{\mu\geq0}L(\mu,x)$")
    ax.legend(loc="center right", fontsize=9)

    fig.suptitle("Q1(c)")
    save(fig, "q1c_primal_function.png")


# ----------------------------------------------------------------------------
# Q2(a)  Saddle point (x*, mu*) = (1, 2)
# ----------------------------------------------------------------------------
X_STAR, MU_STAR = 1.0, 2.0


def q2a():
    Lss = L(MU_STAR, X_STAR)
    print(f"\nQ2(a)(i)   L(mu*, x*) = 1 + 2(1-1) = {Lss}")

    mu = np.linspace(0, 6, 300)
    Lmu = L(mu, X_STAR)
    print(f"Q2(a)(ii)  L(mu, x*) = 1 + mu*0 = 1 for all mu: "
          f"min={Lmu.min()}, max={Lmu.max()}  -> L(mu,x*) <= L(mu*,x*) holds")

    x = np.linspace(-2, 4, 601)
    Lx = L(MU_STAR, x)
    print(f"Q2(a)(iii) max |L(2,x) - ((x-1)^2+1)| = {np.max(np.abs(Lx - ((x-1)**2 + 1))):.1e}; "
          f"min_x L(2,x) = {Lx.min():.4f} at x = {x[np.argmin(Lx)]:.3f}  -> right inequality holds")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(mu, Lmu, "b", lw=2, label=r"$L(\mu, x^*)=1$")
    ax.plot(MU_STAR, Lss, "ro", ms=9, zorder=5, label=r"$L(\mu^*,x^*)=1$")
    ax.set_ylim(0, 3)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu, 1)$")
    ax.set_title(r"$L(\mu,x^*)$ vs $\mu$ (max over $\mu$)")
    ax.legend()

    ax = axes[1]
    ax.plot(x, Lx, "g", lw=2, label=r"$L(\mu^*,x)=(x-1)^2+1$")
    ax.plot(X_STAR, Lss, "ro", ms=9, zorder=5, label=r"$L(\mu^*,x^*)=1$")
    ax.axhline(Lss, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L(2, x)$")
    ax.set_title(r"$L(\mu^*,x)$ vs $x$ (min over $x$)")
    ax.legend()

    fig.suptitle(r"Q2(a)(iv): saddle point at $(x^*,\mu^*)=(1,2)$")
    save(fig, "q2a_saddle_slices.png")

    # Bonus: 3D surface to visualise the saddle
    M, XX = np.meshgrid(np.linspace(0, 4, 80), np.linspace(-1, 3, 80))
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(M, XX, L(M, XX), cmap="coolwarm", alpha=0.75, lw=0)
    ax.plot(mu[mu <= 4], np.full_like(mu[mu <= 4], X_STAR), Lmu[mu <= 4], "b", lw=2.5)
    xs = np.linspace(-1, 3, 100)
    ax.plot(np.full_like(xs, MU_STAR), xs, L(MU_STAR, xs), "g", lw=2.5)
    ax.scatter([MU_STAR], [X_STAR], [Lss], color="r", s=60)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$x$")
    ax.set_zlabel(r"$L(\mu,x)$")
    ax.set_title(r"$L(\mu,x)$ surface with saddle point $(2,1)$")
    save(fig, "q2a_saddle_surface.png")


# ----------------------------------------------------------------------------
# Q2(b)  Numerical check of the chain f(x*) = L(mu*,x*) <= L(mu*,x) = f+mu*h <= f
# ----------------------------------------------------------------------------
def q2b():
    print("\nQ2(b)  chain f(x*) = L(mu*,x*) <= L(mu*,x) = f(x)+mu*h(x) <= f(x)  (feasible x)")
    print(f"  {'x':>5} {'f(x*)':>7} {'L(mu*,x*)':>10} {'L(mu*,x)':>9} {'f+mu*h':>8} {'f(x)':>6}")
    for xi in [1.0, 1.5, 2.0, 3.0]:
        print(f"  {xi:5.1f} {f(X_STAR):7.3f} {L(MU_STAR, X_STAR):10.3f} "
              f"{L(MU_STAR, xi):9.3f} {f(xi) + MU_STAR * h(xi):8.3f} {f(xi):6.3f}")
    print("  h(x*) = 1 - 1 = 0  -> L(mu, x*) = f(x*) + mu*h(x*) stays finite as mu -> inf")


# ----------------------------------------------------------------------------
# Q2(c)  Complementary slackness, and the modified constraint x >= -1
# ----------------------------------------------------------------------------
def q2c():
    print(f"\nQ2(c)  mu* h(x*) = {MU_STAR} * {h(X_STAR)} = {MU_STAR * h(X_STAR)}")

    # Modified problem: h2(x) = -1 - x <= 0, L2 = x^2 + mu(-1 - x)
    # x(mu) = mu/2,  d2(mu) = -mu^2/4 - mu  -> max at mu = 0, d2* = 0
    h2 = lambda x: -1 - x
    d2 = lambda mu: -mu**2 / 4 - mu
    print(f"  x >= -1: unconstrained minimizer x=0 has h2(0) = {h2(0)} < 0 -> feasible, inactive")
    print("  d2(mu) = -mu - mu^2/4 is decreasing on mu >= 0 -> mu* = 0, d2* = 0 = p*")

    mu = np.linspace(0, 4, 300)
    x = np.linspace(-2, 2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(x, x**2, "b", lw=2, label=r"$f(x)=x^2$")
    ax.axvspan(-2, -1, color="r", alpha=0.08, label="infeasible ($x<-1$)")
    ax.axvline(-1, color="r", ls=":", lw=1.5)
    ax.plot(0, 0, "go", ms=9, label=r"$x^*=0$ (interior, $h(x^*)=-1<0$)")
    ax.set_xlabel(r"$x$")
    ax.set_title(r"Modified problem: $\min x^2$ s.t. $x\geq-1$")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(mu, d(mu), "b", lw=2, label=r"original: $d(\mu)=\mu-\mu^2/4$")
    ax.plot(mu, d2(mu), "m", lw=2, label=r"modified: $d(\mu)=-\mu-\mu^2/4$")
    ax.plot(2, 1, "bo", ms=8)
    ax.plot(0, 0, "mo", ms=8)
    ax.annotate(r"$\mu^*=2$", (2, 1), textcoords="offset points", xytext=(5, 6))
    ax.annotate(r"$\mu^*=0$", (0, 0), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$d(\mu)$")
    ax.set_title("Dual functions: active vs inactive constraint")
    ax.legend(fontsize=9)

    fig.suptitle("Q2(c)")
    save(fig, "q2c_inactive_constraint.png")


# ----------------------------------------------------------------------------
# Q3(a)  Weak duality check d(mu) <= p* = 1
# ----------------------------------------------------------------------------
def q3a():
    p_star = f(X_STAR)
    print(f"\nQ3(a)  weak duality: d(mu) <= p* = {p_star}")
    print(f"  {'mu':>3} {'d(mu)':>7} {'p*-d(mu)':>9} {'d<=p*':>6}")
    for mu in [0, 1, 2, 3]:
        print(f"  {mu:3d} {d(mu):7.3f} {p_star - d(mu):9.3f} {str(d(mu) <= p_star):>6}")

    mu = np.linspace(0, 5, 300)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(mu, d(mu), "b", lw=2, label=r"$d(\mu)$")
    ax.axhline(p_star, color="r", ls="--", lw=1.5, label=r"$p^*=1$")
    ax.fill_between(mu, d(mu), p_star, color="orange", alpha=0.2, label="duality gap $p^*-d(\\mu)$")
    pts = np.array([0, 1, 2, 3])
    ax.plot(pts, d(pts), "ko", ms=7, zorder=5)
    for m in pts:
        ax.annotate(f"{d(m):.2f}", (m, d(m)), textcoords="offset points", xytext=(4, -14))
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("value")
    ax.set_title(r"Q3(a): weak duality $d(\mu)\leq p^*$")
    ax.legend()
    save(fig, "q3a_weak_duality.png")


# ----------------------------------------------------------------------------
# Q3(b)  Lines L(mu, x) in mu for x = 0, 1, 2 and their lower envelope
# ----------------------------------------------------------------------------
def q3b():
    mu = np.linspace(0, 5, 500)
    xs = [0, 1, 2]
    lines = np.array([L(mu, xi) for xi in xs])   # mu, 1, 4 - mu
    envelope = lines.min(axis=0)

    print("\nQ3(b)  L(mu,0)=mu,  L(mu,1)=1,  L(mu,2)=4-mu")
    print("  lower envelope = min(mu, 1, 4-mu): pointwise min of affine functions -> concave")
    print(f"  max envelope - d(mu) >= 0 everywhere: {np.all(envelope >= d(mu) - 1e-12)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:blue", "tab:green", "tab:purple"]
    for xi, li, c in zip(xs, lines, colors):
        ax.plot(mu, li, color=c, lw=1.5, ls="--", label=rf"$L(\mu,{xi})$")
    ax.plot(mu, envelope, "k", lw=3, alpha=0.8, label="lower envelope (3 lines)")
    ax.plot(mu, d(mu), "r", lw=2, label=r"true $d(\mu)$ (envelope of all $x$)")
    ax.set_ylim(-1, 4.5)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu,x)$")
    ax.set_title(r"Q3(b): $d(\mu)$ as lower envelope of lines in $\mu$")
    ax.legend(fontsize=9, loc="lower center")
    save(fig, "q3b_lower_envelope.png")


if __name__ == "__main__":
    q1a()
    q1b()
    q1c()
    q2a()
    q2b()
    q2c()
    q3a()
    q3b()
    print(f"\nAll figures written to ./{OUT}/")
