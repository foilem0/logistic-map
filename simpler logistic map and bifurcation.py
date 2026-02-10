import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# change this stuff
a_eval = 3.6  # single 'a' value to analyze & plot orbit for
x0 = 0.3  # initial condition
transient = 1000  # transient iterations to discard for orbit analysis
iterations = 400  # number of post-transient points to collect for orbit plotting / period detection
n_show = 100  # how many of the post-transient points to show in the orbit plot (<= iterations)

# bifurcation sweep config
a_min, a_max = 2.5, 4.0
a_points = 2000  # number of 'a' values to sweep
bifurc_transient = 1200  # transient iterations per 'a' in the bifurcation diagram
bifurc_samples = (
    200  # how many post-transient points to record per 'a' (plotted vertically)
)

# Safety checks
a_points = int(max(10, min(2000, a_points)))
bifurc_samples = int(max(1, min(500, bifurc_samples)))
transient = int(max(0, min(200000, transient)))
bifurc_transient = int(max(0, min(200000, bifurc_transient)))
iterations = int(max(1, min(5000, iterations)))
n_show = int(max(1, min(iterations, n_show)))


def logistic_map(x, a):
    return a * x * (1 - x)


def simulate_orbit(a, x0=0.5, transient=1000, iterations=200):
    x = float(x0)
    for _ in range(transient):
        x = logistic_map(x, a)
    orbit = np.empty(iterations)
    for i in range(iterations):
        x = logistic_map(x, a)
        orbit[i] = x
    return orbit


def detect_period(orbit, candidate_periods=(1, 2, 4, 8, 16), tol=1e-6):
    N = len(orbit)
    # suffix length (at least a few multiples of max period)
    suffix_len = min(N, max(50, N // 2))
    start = N - suffix_len
    for p in candidate_periods:
        if p >= suffix_len:
            continue
        ok = True
        for i in range(start, N - p):
            if abs(orbit[i] - orbit[i + p]) > tol:
                ok = False
                break
        if ok:
            return p
    return None


orbit = simulate_orbit(a_eval, x0=x0, transient=transient, iterations=iterations)
detected = detect_period(orbit, candidate_periods=(1, 2, 4, 8, 16), tol=1e-6)

print(f"Analyzed a = {a_eval}, x0 = {x0}")
print(
    f"Transient discarded: {transient} iters; Post-transient iterations collected: {iterations}"
)
print(
    "Detected period:",
    ("period-" + str(detected))
    if detected is not None
    else "no small period detected (likely chaotic or higher period)",
)

fig1, ax1 = plt.subplots(figsize=(10, 4.5))
ns = np.arange(1, n_show + 1)
ys = orbit[:n_show]
ax1.plot(ns, ys, marker="o", linestyle="-")
yticks = np.arange(1.0, -0.001, -0.2)
ax1.set_yticks(yticks)
ax1.set_ylim(1.0, 0.0)  # invert
ax1.set_xlabel("Iteration n (post-transient)")
ax1.set_ylabel(r"$x_n$ (state)")
ax1.set_title(
    f'Logistic map orbit (a={a_eval}, x0={x0}) — detected: {("period-"+str(detected)) if detected is not None else "chaos/higher period"}'
)
ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

sample_text = "\n".join(
    [f"n={i+1}: {val:.6f}" for i, val in enumerate(ys[-min(6, len(ys)) :])]
)
ax1.text(
    0.98,
    0.02,
    sample_text,
    transform=ax1.transAxes,
    fontsize=9,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round", alpha=0.12),
)

out_orbit = "logistic_orbit_analysis.png"
plt.tight_layout()
plt.savefig(out_orbit, dpi=300)
plt.show()

print(f"Saved orbit plot to: {out_orbit}")

a_values = np.linspace(a_min, a_max, a_points)
a_plot = []
x_plot = []

print("Starting bifurcation sweep — this may take a little while...")
for idx, a in enumerate(a_values):
    x = float(x0)
    for _ in range(bifurc_transient):
        x = logistic_map(x, a)
    for _ in range(bifurc_samples):
        x = logistic_map(x, a)
        a_plot.append(a)
        x_plot.append(x)
    if (idx + 1) % max(1, (a_points // 10)) == 0:
        pct = int(100 * (idx + 1) / a_points)
        print(f"  done {pct}% of a sweep")

a_plot = np.array(a_plot)
x_plot = np.array(x_plot)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(a_plot, x_plot, marker=".", linestyle="None", markersize=0.8)
ax2.set_xlabel("a (control parameter)")
ax2.set_ylabel(r"$x$ (post-transient values)")
ax2.set_title(
    f"Bifurcation diagram: a in [{a_min},{a_max}] — transient {bifurc_transient}, samples per a {bifurc_samples}"
)
ax2.set_xlim(a_min, a_max)
ax2.set_ylim(0, 1)
ax2.grid(False)

out_bif = "logistic_bifurcation.png"
plt.tight_layout()
plt.savefig(out_bif, dpi=220)
plt.show()

print(f"Saved bifurcation diagram to: {out_bif}")

# save csv of orbit values
csv_path = "logistic_orbit_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "x_n"])
    for i, val in enumerate(orbit, start=1):
        writer.writerow([i, f"{val:.12f}"])
print(f"Saved post-transient orbit values to CSV: {csv_path}")