import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm


# ------------------- Parameter initialization ------------------- #
Sn_initial = 26                       # Initial Sn concentration in the melt (ppm)
D_mineral_melt = 0.157                # Mineral–melt partition coefficient for Sn
D_fluid_melt = 7                      # Fluid–melt partition coefficient for Sn
P = 146                               # Pressure (MPa)
delta_Sn_initial = -0.236             # Initial Sn isotopic composition (δ units)
alpha1 = 0.9999                       # Fractionation factor: mineral–melt
alpha2 = 1.0001                       # Fractionation factor: fluid–melt

# Observational dataset
data = pd.DataFrame({
    "X_content": [26.0, 28.6, 33.0, 27.6, 34.3, 28.5, 28.8, 26.7,
                  25.7, 25.9, 29.6, 37.0, 24.7, 155.0, 39.9, 32.0],
    "Y_isotope": [-0.236, -0.447, -0.297, -0.297, -0.106, 0.285,
                  -0.508, 1.073, 0.152, 0.126, -0.279, -0.846,
                  0.015, -0.619, -0.107, -0.145]
})


# ------------------- Water exsolution parameter (Phi_H2O) ------------------- #
Phi_H2O = -0.01 * (
    P * 0.02859
    - P**1.5 * 0.001495
    + P**2 * 0.00002702
    + P**0.5 * 0.257
)

# Effective bulk partition coefficient and isotopic fractionation parameter
lambda1 = Phi_H2O * (D_mineral_melt - D_fluid_melt) + D_mineral_melt
lambda2 = Phi_H2O * (alpha1 - alpha2) + alpha2


# ------------------- Sn–δ evolution curves ------------------- #
F_value = np.linspace(0, 1, 1000)     # Residual melt fraction
f_value = F_value ** lambda1

# Sn concentration evolution
Sn_residual = Sn_initial * F_value ** (lambda1 - 1)
Sn_inst_solid = Sn_initial * D_mineral_melt * F_value ** (lambda1 - 1)
Sn_inst_fluid = Sn_initial * D_fluid_melt * F_value ** (lambda1 - 1)

# Sn isotopic evolution
delta_Sn_residual = (delta_Sn_initial + 1000) * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_solid = (delta_Sn_initial + 1000) * alpha1 * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_fluid = (delta_Sn_initial + 1000) * alpha2 * f_value ** (lambda2 - 1) - 1000


# ------------------- Curve fitting ------------------- #
def fit_func(X, a, X0, Y0):
    """
    Power-law fitting function for the Sn–δ relationship
    """
    return (Y0 + 1000) * (X / X0)**a - 1000


X0, Y0 = Sn_initial, delta_Sn_initial

# Fit the exponent a
popt, _ = curve_fit(
    lambda X, a: fit_func(X, a, X0, Y0),
    data["X_content"],
    data["Y_isotope"],
    p0=[1]
)

a_fit = popt[0]

# Fitted values and uncertainty estimate
Y_fit = fit_func(data["X_content"], a_fit, X0, Y0)
error_std = np.std(data["Y_isotope"] - Y_fit)

X_grid = np.linspace(
    min(data["X_content"]),
    max(data["X_content"]) + 10,
    1000
)
Y_fit_grid = fit_func(X_grid, a_fit, X0, Y0)


# ------------------- Monte Carlo simulation ------------------- #
np.random.seed(123)
num_sim = 10000

# Fractionation factor ranges
alpha1_range = (0.995, 1.005)
alpha2_range = (0.995, 1.005)

records = []
X1, Y1 = Sn_initial, delta_Sn_initial
X_grid2 = np.linspace(X1, 180, 1000)

for i in range(num_sim):
    a1 = np.random.uniform(*alpha1_range)
    a2 = np.random.uniform(*alpha2_range)

    lambda2_sim = Phi_H2O * (a1 - a2) + a2

    Y_sim = (Y1 + 1000) * (data["X_content"] / X1)**(
        lambda1 * (lambda2_sim - 1) / (lambda1 - 1)
    ) - 1000

    mse = np.mean((Y_sim - data["Y_isotope"])**2)
    records.append([a1, a2, mse])

# Select the top 100 best-fitting simulations
top100 = (
    pd.DataFrame(records, columns=["alpha1", "alpha2", "mse"])
    .sort_values("mse")
    .head(100)
)


# ------------------- Combined plot (all curves) ------------------- #
plt.figure(figsize=(11, 7))

# --- 1. Monte Carlo curves (background, RdBu colormap) --- #
cmap = cm.get_cmap("RdBu", 100)
colors = cmap(np.linspace(0, 1, 100))

for i in range(100):
    a1, a2 = top100.iloc[i][["alpha1", "alpha2"]]
    lambda2_sim = Phi_H2O * (a1 - a2) + a2

    Y_sim = (Y1 + 1000) * (X_grid2 / X1)**(
        lambda1 * (lambda2_sim - 1) / (lambda1 - 1)
    ) - 1000

    plt.plot(X_grid2, Y_sim, color=colors[i], lw=1, alpha=0.9)

# --- 2. Main Sn–δ evolution curves --- #
plt.plot(Sn_residual, delta_Sn_residual,
         color='black', lw=2, label='[Sn] residual melt')
plt.plot(Sn_inst_solid, delta_Sn_inst_solid,
         color='darkorange', lw=2, label='[Sn] instantaneous solid')
plt.plot(Sn_inst_fluid, delta_Sn_inst_fluid,
         color='green', lw=2, label='[Sn] instantaneous fluid')

# --- 3. Data points and best-fit curve --- #
plt.scatter(data["X_content"], data["Y_isotope"],
            color='blue', s=40, label='Observed data')

plt.plot(X_grid, Y_fit_grid,
         color='red', lw=2.5, label='Best-fit curve')

plt.fill_between(
    X_grid,
    Y_fit_grid - error_std,
    Y_fit_grid + error_std,
    color='red', alpha=0.15, label='Fit uncertainty'
)

# --- Figure settings --- #
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.xlabel("Sn concentration (ppm)")
plt.ylabel("δ124Sn (‰)")
plt.title("Combined Sn–δSn Model: Rayleigh Fractionation + Monte Carlo Simulation")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.tight_layout()

# --- Save as editable SVG vector graphic --- #
plt.rcParams['svg.fonttype'] = 'none'   # Keep text editable in SVG
output_path = r"D:\Sn_isotope_plot.svg"
plt.savefig(output_path, format="svg")

plt.show()


# ------------------- Export best-fit Monte Carlo parameters ------------------- #
output_dir = r"D:"
os.makedirs(output_dir, exist_ok=True)

# Save results to Excel
alpha_output_xlsx = os.path.join(output_dir, "top100_alpha.xlsx")
top100[["alpha1", "alpha2", "mse"]].to_excel(alpha_output_xlsx, index=False)
