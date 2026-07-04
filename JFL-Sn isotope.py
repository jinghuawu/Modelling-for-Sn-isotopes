import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm


# ------------------- Parameter Initialization ------------------- #
Sn_initial = 16.2
D_mineral_melt = 0.141
D_fluid_melt = 7
P = 150
delta_Sn_initial = 0.604
alpha1 = 0.9999
alpha2 = 1.0001

data = pd.DataFrame({
    "X_content": [26.10, 34.80, 25.10, 20.10, 22.40, 23.40, 24.50, 16.20, 24.00, 20.10, 24.00, 16.40, 59.70, 54.10, 27.90],
    "Y_isotope": [0.469, 0.300, 0.239, 0.306, 0.470, 0.469, 0.473, 0.604, 0.180, 0.296, 0.325, 0.408, 0.254, -0.122, 0.449]
})

# ------------------- Calculate Phi_H2O ------------------- #
Phi_H2O = -0.01 * (P * 0.02859 - P**1.5 * 0.001495 + P**2 * 0.00002702 + P**0.5 * 0.257)

lambda1 = Phi_H2O * (D_mineral_melt - D_fluid_melt) + D_mineral_melt
lambda2 = Phi_H2O * (alpha1 - alpha2) + alpha2

# ------------------- Sn–δ Curves ------------------- #
F_value = np.linspace(0, 1, 1000)
f_value = F_value ** lambda1

Sn_residual = Sn_initial * F_value ** (lambda1 - 1)
Sn_inst_solid = Sn_initial * D_mineral_melt * F_value ** (lambda1 - 1)
Sn_inst_fluid = Sn_initial * D_fluid_melt * F_value ** (lambda1 - 1)

delta_Sn_residual = (delta_Sn_initial + 1000) * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_solid = (delta_Sn_initial + 1000) * alpha1 * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_fluid = (delta_Sn_initial + 1000) * alpha2 * f_value ** (lambda2 - 1) - 1000

# ------------------- Curve Fitting ------------------- #
def fit_func(X, a, X0, Y0):
    return (Y0 + 1000) * (X / X0) ** a - 1000

X0, Y0 = Sn_initial, delta_Sn_initial

popt, _ = curve_fit(
    lambda X, a: fit_func(X, a, X0, Y0),
    data["X_content"],
    data["Y_isotope"],
    p0=[1]
)

a_fit = popt[0]

Y_fit = fit_func(data["X_content"], a_fit, X0, Y0)
error_std = np.std(data["Y_isotope"] - Y_fit)

X_grid = np.linspace(min(data["X_content"]), max(data["X_content"]) + 10, 1000)
Y_fit_grid = fit_func(X_grid, a_fit, X0, Y0)

# ------------------- Monte Carlo Simulation ------------------- #
np.random.seed(123)
num_sim = 5000

alpha1_range = (0.995, 1.005)
alpha2_range = (0.995, 1.005)

records = []

X1, Y1 = Sn_initial, delta_Sn_initial
X_grid2 = np.linspace(X1, 180, 1000)

for i in range(num_sim):
    a1 = np.random.uniform(*alpha1_range)
    a2 = np.random.uniform(*alpha2_range)

    lambda2_sim = Phi_H2O * (a1 - a2) + a2

    Y_sim = (Y1 + 1000) * (data["X_content"] / X1) ** (
        lambda1 * (lambda2_sim - 1) / (lambda1 - 1)
    ) - 1000

    mse = np.mean((Y_sim - data["Y_isotope"]) ** 2)

    records.append([a1, a2, mse])

top100 = (
    pd.DataFrame(records, columns=["alpha1", "alpha2", "mse"])
    .sort_values("mse")
    .head(100)
)

# ------------------- Combined Plot ------------------- #
plt.figure(figsize=(11, 7))

# --- Plot the top 100 Monte Carlo curves as background (RdBu colormap) --- #
cmap = cm.get_cmap("RdBu", 100)
colors = cmap(np.linspace(0, 1, 100))

for i in range(100):
    a1, a2 = top100.iloc[i][["alpha1", "alpha2"]]

    lambda2_sim = Phi_H2O * (a1 - a2) + a2

    Y_sim = (Y1 + 1000) * (X_grid2 / X1) ** (
        lambda1 * (lambda2_sim - 1) / (lambda1 - 1)
    ) - 1000

    plt.plot(X_grid2, Y_sim, color=colors[i], lw=1, alpha=0.9)

# --- Plot the three theoretical Sn–δ curves --- #
plt.plot(Sn_residual, delta_Sn_residual,
         color='black', lw=2, label='Residual melt')

plt.plot(Sn_inst_solid, delta_Sn_inst_solid,
         color='darkorange', lw=2, label='Instantaneous solid')

plt.plot(Sn_inst_fluid, delta_Sn_inst_fluid,
         color='green', lw=2, label='Instantaneous fluid')

# --- Plot observed data and best-fit curve --- #
plt.scatter(
    data["X_content"],
    data["Y_isotope"],
    color='blue',
    s=40,
    label='Observed data'
)

plt.plot(
    X_grid,
    Y_fit_grid,
    color='red',
    lw=2.5,
    label='Best-fit curve'
)

plt.fill_between(
    X_grid,
    Y_fit_grid - error_std,
    Y_fit_grid + error_std,
    color='red',
    alpha=0.15,
    label='±1σ fit uncertainty'
)

# ------------------- Figure Formatting ------------------- #
plt.xlim(0, 200)
plt.ylim(-1, 1)

plt.xlabel("Sn concentration (ppm)")
plt.ylabel("δamuSn")

plt.title("Sn Isotope Modeling: Theoretical Curves, Best Fit, and Monte Carlo Simulation")

plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.tight_layout()

# ------------------- Save as SVG Vector Graphic ------------------- #
plt.rcParams['svg.fonttype'] = 'none'   # Keep text editable in the SVG

output_path = r"D:\Data\Sn_isotope_plot.svg"
plt.savefig(output_path, format="svg")

plt.show()

# ------------------- Export the Top 100 Parameter Sets ------------------- #
output_dir = r"D:\Data"
os.makedirs(output_dir, exist_ok=True)

alpha_output_xlsx = os.path.join(output_dir, "top100_alpha.xlsx")

top100[["alpha1", "alpha2", "mse"]].to_excel(
    alpha_output_xlsx,
    index=False
)
