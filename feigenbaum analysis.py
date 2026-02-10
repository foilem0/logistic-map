import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root


def logistic_map(x, a):
    # logistic map function: x_{n+1} = a * x_n * (1 - x_n)
    return a * x * (1 - x)


def iterate_map(a, x0=0.3, n_transient=2000, n_sample=500):

    # - a: control parameter
    # - x0: initial condition
    # - n_transient: number of iterations to discard (transient behavior)
    # - n_sample: number of iterations to sample after transient

    x = x0

    for _ in range(n_transient):
        x = logistic_map(x, a)

    samples = []
    for _ in range(n_sample):
        x = logistic_map(x, a)
        samples.append(x)

    return np.array(samples)


def detect_period(samples, period, tolerance=1e-6):
    if len(samples) < 2 * period:
        return False

    for i in range(len(samples) - period):
        if abs(samples[i] - samples[i + period]) > tolerance:
            return False

    return True


def multiplier_for_period(a, period, x0=0.3, n_transient=5000, n_sample=None):
    if n_sample is None:
        n_sample = max(2 * period, 2 * 128)

    samples = iterate_map(a, x0=x0, n_transient=n_transient, n_sample=n_sample)

    if len(samples) < period:
        return None

    orbit = samples[-period:]

    unique_vals = np.unique(np.round(orbit, 10))
    if len(unique_vals) < period:
        return None

    # derivative of logistic map a*(1-2x)
    derivs = a * (1 - 2 * orbit)
    multiplier = np.prod(derivs)
    return multiplier


def find_bifurcation_point(period, a_low, a_high, tolerance=1e-8):
    def multiplier_plus_one(a):
        m = multiplier_for_period(a, period, n_transient=6000)
        if m is None:
            raise ValueError(
                "Could not detect a clean period-p orbit at a={:.8f}".format(a)
            )
        return m + 1.0

    def f_p_and_deriv(x, a, p):
        val = x
        deriv = 1.0
        for _ in range(p):
            deriv *= a * (1 - 2 * val)
            val = logistic_map(val, a)
        return val, deriv

    try:
        a_guess = (a_low + a_high) / 2.0
        # get a decent x guess by iterating the map at the guess parameter
        xs = iterate_map(a_guess, n_transient=8000, n_sample=800)
        x_guess = float(xs[-1])

        def two_eqs(vars):
            x_var, a_var = vars
            fp, dfdp = f_p_and_deriv(x_var, a_var, period)
            return [fp - x_var, dfdp + 1.0]

        sol = root(two_eqs, [x_guess, a_guess], tol=1e-12)
        if sol.success:
            a_candidate = float(sol.x[1])
            if a_low <= a_candidate <= a_high:
                return a_candidate
    except Exception:
        pass

    # brentq method
    try:

        def safe_mult_plus_one(a):
            try:
                return multiplier_plus_one(a)
            except ValueError:
                return np.nan

        f_low = safe_mult_plus_one(a_low)
        f_high = safe_mult_plus_one(a_high)

        if np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0:
            a_root = brentq(
                safe_mult_plus_one, a_low, a_high, xtol=tolerance, maxiter=100
            )
            return a_root
        else:
            raise RuntimeError(
                "bad bracketing for brentq, falling back to heuristic bisection"
            )
    except Exception:
        a_mid = (a_low + a_high) / 2
        max_iterations = 80

        for _ in range(max_iterations):
            a_mid = (a_low + a_high) / 2

            if a_high - a_low < tolerance:
                break

            samples_low = iterate_map(a_low, n_transient=4000, n_sample=400)
            samples_mid = iterate_map(a_mid, n_transient=4000, n_sample=400)

            is_period_p_at_low = detect_period(samples_low, period, tolerance=1e-7)
            is_period_2p_at_mid = detect_period(samples_mid, 2 * period, tolerance=1e-7)

            if is_period_2p_at_mid and is_period_p_at_low:
                a_high = a_mid
            else:
                # otherwise, the root is in (a_mid, a_high)
                a_low = a_mid

        return a_mid


def compute_bifurcation_parameters(max_n=7):

    bifurcation_params = []

    # some known values and search ranges found from literature
    search_ranges = [
        (2.8, 3.2),  # a_1 around 3.0
        (3.4, 3.5),  # a_2 around 3.449
        (3.54, 3.55),  # a_3 around 3.544
        (3.564, 3.565),  # a_4 around 3.5644
        (3.5687, 3.5688),  # a_5 around 3.56876
        (3.56969, 3.5697),  # a_6 around 3.56969
        (3.56989, 3.5699),  # a_7 around 3.56989
    ]

    for n in range(1, min(max_n + 1, len(search_ranges) + 1)):
        print(f"Computing a_{n}...")
        period = 2 ** (n - 1)
        a_low, a_high = search_ranges[n - 1]

        if n > 1:
            a_low = bifurcation_params[-1]

        a_n = find_bifurcation_point(period, a_low, a_high)
        bifurcation_params.append(a_n)
        print(f"  a_{n} ≈ {a_n:.12f}")

    return np.array(bifurcation_params)


def compute_feigenbaum_ratios(bifurcation_params):

    # Feigenbaum ratios δ_n = (a_{n-1} - a_{n-2}) / (a_n - a_{n-1})
    ratios = []

    for n in range(2, len(bifurcation_params)):
        delta_n = (bifurcation_params[n - 1] - bifurcation_params[n - 2]) / (
            bifurcation_params[n] - bifurcation_params[n - 1]
        )
        ratios.append(delta_n)

    return np.array(ratios)


def create_bifurcation_diagram(a_min=2.8, a_max=4.0, n_points=3000):
    a_values = np.linspace(a_min, a_max, n_points)

    fig, ax = plt.subplots(figsize=(12, 8))

    for a in a_values:
        samples = iterate_map(a, n_transient=2000, n_sample=100)
        unique_samples = np.unique(np.round(samples, 8))
        ax.plot([a] * len(unique_samples), unique_samples, "k,", markersize=0.5)

    ax.set_xlabel("Control Parameter a", fontsize=12)
    ax.set_ylabel("x (post-transient values)", fontsize=12)
    ax.set_title(
        "Bifurcation Diagram for the Logistic Map", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_convergence(ratios):
    feigenbaum_constant = 4.669201609102990

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_values = np.arange(3, 3 + len(ratios))
    ax1.plot(n_values, ratios, "bo-", linewidth=2, markersize=8, label="Computed δ_n")
    ax1.axhline(
        y=feigenbaum_constant,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"δ ≈ {feigenbaum_constant:.6f}",
    )
    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("δ_n", fontsize=12)
    ax1.set_title("Convergence of Feigenbaum Ratios", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs n
    errors = np.abs(ratios - feigenbaum_constant)
    ax2.semilogy(n_values, errors, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("n", fontsize=12)
    ax2.set_ylabel("|δ_n - δ|", fontsize=12)
    ax2.set_title("Absolute Error in Feigenbaum Ratios", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("LOGISTIC MAP BIFURCATION ANALYSIS")
    print("=" * 60)

    print("\n1. Computing bifurcation parameters a_n...")
    print("-" * 60)
    bifurcation_params = compute_bifurcation_parameters(max_n=7)

    print("\n2. Computing Feigenbaum ratios δ_n...")
    print("-" * 60)
    ratios = compute_feigenbaum_ratios(bifurcation_params)

    print("\n3. RESULTS TABLE")
    print("-" * 60)
    print(f"{'n':<5} {'a_n':<20} {'δ_n':<20}")
    print("-" * 60)

    for i, a_n in enumerate(bifurcation_params, start=1):
        if i == 1:
            print(f"{i:<5} {a_n:<20.12f} {'—':<20}")
        elif i == 2:
            print(f"{i:<5} {a_n:<20.12f} {'—':<20}")
        else:
            delta_n = ratios[i - 3]
            print(f"{i:<5} {a_n:<20.12f} {delta_n:<20.12f}")

    print("-" * 60)
    print(f"\nFeigenbaum constant δ ≈ 4.669201609102990")
    print(f"Final computed ratio: δ_{{{len(ratios)+2}}} ≈ {ratios[-1]:.12f}")
    print(f"Error: {abs(ratios[-1] - 4.669201609102990):.2e}")

    print("\n4. Creating bifurcation diagram...")
    fig1 = create_bifurcation_diagram(a_min=2.8, a_max=4.0)
    plt.savefig("bifurcation_diagram.png", dpi=500, bbox_inches="tight")

    print("5. Creating convergence plots...")
    fig2 = plot_convergence(ratios)
    plt.savefig("convergence_plot.png", dpi=500, bbox_inches="tight")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  - bifurcation_diagram.png")
    print("  - convergence_plot.png")

    plt.show()
