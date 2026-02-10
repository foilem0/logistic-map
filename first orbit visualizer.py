import argparse
import numpy as np
import matplotlib.pyplot as plt

a = 4.0  # control parameter
x0 = 0.3  # initial state x_0
n_iter = 50  # total number of iterations to compute
n_show = 50  # how many of the computed n-states to show on the image (<= n_iter)

n_iter = int(max(1, min(5000, n_iter)))
n_show = int(max(1, min(n_iter, n_show)))


def compute_logistic_orbit(a, x0, n_iter):
    xs = np.empty(n_iter + 1)
    xs[0] = float(x0)
    for n in range(n_iter):
        xs[n + 1] = a * xs[n] * (1 - xs[n])
    return xs


plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14})


def plot_logistic_orbit(
    a, x0, n_iter, n_show, ax=None, title_prefix="Logistic map orbit"
):
    xs = compute_logistic_orbit(a, x0, n_iter)
    ns = np.arange(n_show + 1)
    ys = xs[: n_show + 1]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    ax.plot(
        ns,
        ys,
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        color="#1f77b4",
        alpha=0.95,
    )

    yticks = np.linspace(0.0, 1.0, 6)
    ax.set_yticks(yticks)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Iteration n (discrete time step)")
    ax.set_ylabel(r"x_n (state)")
    ax.set_title(
        f"{title_prefix}: a = {a}, x0 = {x0}, shown = {n_show} (computed {n_iter})"
    )

    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.minorticks_on()

    # remove top/right spines for a more modern look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # bring up a few recent values on the right for reference
    sample_text = "\n".join(
        [
            f"n={i}: {vals:.6f}"
            for i, vals in zip(
                range(max(0, n_show - 5), n_show + 1), ys[max(0, n_show - 5) :]
            )
        ]
    )
    ax.text(
        0.98,
        0.02,
        sample_text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round", facecolor="#ffffff", alpha=0.8, edgecolor="#cccccc"
        ),
    )

    # highlight the last point so its easy to spot on the plot
    if len(ns) > 0:
        ax.scatter(
            ns[-1],
            ys[-1],
            s=80,
            color="#d62728",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

    if created_fig:
        plt.tight_layout()
        return fig, ax
    return None, ax


def save_and_show(
    filename_prefix="logistic_orbit", dpi=200, save_png=True, save_pdf=False
):
    if save_png:
        png = f"{filename_prefix}.png"
        plt.savefig(png, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {png}")
    if save_pdf:
        pdf = f"{filename_prefix}.pdf"
        plt.savefig(pdf, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {pdf}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="plot the logistic map orbit (improved visualization)"
    )
    parser.add_argument(
        "--a", type=float, default=a, help="control parameter (default: %(default)s)"
    )
    parser.add_argument(
        "--x0", type=float, default=x0, help="initial state x0 (default: %(default)s)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=n_iter,
        help="how many iter to compute (default: %(default)s)",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=n_show,
        help="how many iterations to show (<= n-iter) (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default="logistic_orbit", help="output filename prefix (no extension)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="image DPI when saving (default: %(default)s)",
    )
    parser.add_argument("--no-png", action="store_true", help="dont save a png file")
    parser.add_argument("--pdf", action="store_true", help="also save a PDF")

    args = parser.parse_args()

    fig, ax = plot_logistic_orbit(a=args.a, x0=args.x0, n_iter=n_iter, n_show=n_show)
    save_and_show(
        filename_prefix=args.out,
        dpi=args.dpi,
        save_png=not args.no_png,
        save_pdf=args.pdf,
    )
