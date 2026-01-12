import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm



# ------------------- 参数初始化 ------------------- #
Sn_initial = 26
D_mineral_melt = 0.157
D_fluid_melt = 7
P = 146
delta_Sn_initial = -0.236
alpha1 = 0.9999
alpha2 = 1.0001

data = pd.DataFrame({
"X_content": [26.0, 28.6, 33.0, 27.6, 34.3, 28.5, 28.8, 26.7, 25.7, 25.9, 29.6, 37.0, 24.7, 155.0, 39.9, 32.0],
"Y_isotope": [-0.236, -0.447, -0.297, -0.297, -0.106, 0.285, -0.508, 1.073, 0.152, 0.126, -0.279, -0.846, 0.015, -0.619, -0.107, -0.145]
})

# ------------------- Phi_H2O ------------------- #
Phi_H2O = -0.01 * (P * 0.02859 - P**1.5 * 0.001495 + P**2 * 0.00002702 + P**0.5 * 0.257)

lambda1 = Phi_H2O * (D_mineral_melt - D_fluid_melt) + D_mineral_melt
lambda2 = Phi_H2O * (alpha1 - alpha2) + alpha2

# ------------------- Sn–δ 曲线 ------------------- #
F_value = np.linspace(0, 1, 1000)
f_value = F_value ** lambda1

Sn_residual = Sn_initial * F_value ** (lambda1 - 1)
Sn_inst_solid = Sn_initial * D_mineral_melt * F_value ** (lambda1 - 1)
Sn_inst_fluid = Sn_initial * D_fluid_melt * F_value ** (lambda1 - 1)

delta_Sn_residual = (delta_Sn_initial + 1000) * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_solid = (delta_Sn_initial + 1000) * alpha1 * f_value ** (lambda2 - 1) - 1000
delta_Sn_inst_fluid = (delta_Sn_initial + 1000) * alpha2 * f_value ** (lambda2 - 1) - 1000

# ------------------- 拟合 ------------------- #
def fit_func(X, a, X0, Y0):
    return (Y0 + 1000) * (X / X0)**a - 1000

X0, Y0 = Sn_initial, delta_Sn_initial
popt, _ = curve_fit(lambda X, a: fit_func(X, a, X0, Y0),
                    data["X_content"], data["Y_isotope"], p0=[1])
a_fit = popt[0]

Y_fit = fit_func(data["X_content"], a_fit, X0, Y0)
error_std = np.std(data["Y_isotope"] - Y_fit)

X_grid = np.linspace(min(data["X_content"]), max(data["X_content"]) + 10, 1000)
Y_fit_grid = fit_func(X_grid, a_fit, X0, Y0)

# ------------------- 蒙特卡洛模拟 ------------------- #
np.random.seed(123)
num_sim = 10000
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
        lambda1*(lambda2_sim - 1)/(lambda1 - 1)
    ) - 1000
    mse = np.mean((Y_sim - data["Y_isotope"])**2)
    records.append([a1, a2, mse])

top100 = pd.DataFrame(records, columns=["alpha1", "alpha2", "mse"]).sort_values("mse").head(100)

# ------------------- 合并绘图（所有曲线） ------------------- #
plt.figure(figsize=(11, 7))

# --- 1. 先绘制 Monte-Carlo 的 50 条曲线（作为背景，RdBu） --- #
cmap = cm.get_cmap("RdBu", 100)
colors = cmap(np.linspace(0, 1, 100))

for i in range(100):
    a1, a2 = top100.iloc[i][["alpha1", "alpha2"]]
    lambda2_sim = Phi_H2O * (a1 - a2) + a2
    Y_sim = (Y1 + 1000) * (X_grid2 / X1)**(
        lambda1*(lambda2_sim - 1)/(lambda1 - 1)
    ) - 1000

    plt.plot(X_grid2, Y_sim, color=colors[i], lw=1, alpha=0.9)

# --- 2. 绘制 Sn–δ 的 3 条主曲线 --- #
plt.plot(Sn_residual, delta_Sn_residual, color='black', lw=2, label='[Sn] residual')
plt.plot(Sn_inst_solid, delta_Sn_inst_solid, color='darkorange', lw=2, label='[Sn] inst. solid')
plt.plot(Sn_inst_fluid, delta_Sn_inst_fluid, color='green', lw=2, label='[Sn] inst. fluid')

# --- 3. 数据点 + 拟合曲线 --- #
plt.scatter(data["X_content"], data["Y_isotope"], color='blue', s=40, label="Data")
plt.plot(X_grid, Y_fit_grid, color='red', lw=2.5, label='Best-fit curve')

plt.fill_between(X_grid,
                 Y_fit_grid - error_std,
                 Y_fit_grid + error_std,
                 color='red', alpha=0.15, label='Fit error band')

# --- 图形设置 --- #
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.xlabel("Sn concentration (ppm)")
plt.ylabel("δamuSn")
plt.title("Combined Plot: Sn–δ Curves + Data + Fit + Monte-Carlo (RdBu)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.tight_layout()

# --- 保存为 SVG 矢量图 --- #
plt.rcParams['svg.fonttype'] = 'none'  # 保持文字为可编辑文本
output_path = r"G:\RData\Sn_isotope_plot.svg"
plt.savefig(output_path, format="svg")

plt.show()

# --- 保存前 100 条最优 alpha1 / alpha2 参数 --- #
output_dir = r"G:\RData"
os.makedirs(output_dir, exist_ok=True)

# 保存为 Excel
alpha_output_xlsx = os.path.join(output_dir, "top100_alpha.xlsx")
top100[["alpha1", "alpha2", "mse"]].to_excel(alpha_output_xlsx, index=False)