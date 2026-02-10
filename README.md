# Logistic Map & Bifurcation Analysis

This project explores the dynamics of the Logistic Map

These scripts can visualize orbits, map out bifurcation diagrams, and numerically calculate the Feigenbaum Constant ($\delta$).

## Features

- Track the discrete-time evolution of states ($x_n$) for specific control parameters.
- Automated detection of period-1, 2, 4, 8, and 16 orbits.
- High-resolution generation of the bifurcation diagram to visualize the period-doubling route to chaos.
- Root-finding to calculate bifurcation points and verify their convergence to the Feigenbaum constant.

---

## Math

The recursive logistic map equation:

$$x_{n+1} = a x_n (1 - x_n)$$

Where:

- $x_n$ is a value between 0 and 1 representing the state at time $n$.
- $a$ is the control parameter (usually between 0 and 4).

As $a$ increases, the system moves from a stable fixed point to periodic oscillations (bifurcations), and eventually into chaos.

---

## Project Structure

### 1. `first_orbit_visualizer.py`

Plots the trajectory of $x$ over a series of iterations. Shows how a system settles into a steady state or bounces between values.

### 2. `simpler_logistic_map_and_bifurcation.py`

Does two things

1. Analyzes the long-term "steady state" of the system after discarding transients.
2. Generates a bifurcation diagram by sweeping $a$ from 2.5 to 4.0.

### 3. `feigenbaum_analysis.py`

Uses `scipy.optimize` to find the exact values of $a$ where the system bifurcates ($a_1, a_2, \dots, a_n$).

- Calculates the ratio of the distances between successive bifurcation points to approximate the Feigenbaum constant $\delta \approx 4.6692$.

---

## Installation and Usage

### Prerequisites

You will need Python 3.x and the following libraries:

Bash

```
pip install numpy matplotlib scipy
```

### Running the Visualizers

To see a single orbit with a control parameter of 3.8:

Bash

```
python "first orbit visualizer.py" --a 3.8 --n-iter 100
```

To run the full Feigenbaum convergence analysis:

Bash

```
python "feigenbaum analysis.py"
```

---

## Results and Insights

The analysis confirms the universal nature of the Feigenbaum constant. Even with the simple quadratic form of the logistic map, the ratio of distances between bifurcations converges rapidly:

| n   | Bifurcation Point (an​) | Ratio (δn​) |
| --- | ----------------------- | ----------- |
| 1   | 3.000000                | —           |
| 2   | 3.449490                | —           |
| 3   | 3.544090                | 4.7514...   |
| 4   | 3.564407                | 4.6562...   |
