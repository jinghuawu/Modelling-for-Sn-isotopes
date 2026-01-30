import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123)  # Set random seed for reproducibility

# Define Monte Carlo iteration numbers
n_values = [1000, 5000, 10000, 50000, 100000, 500000,
            1000000, 5000000, 10000000, 50000000, 100000000]

valid_results_all = pd.DataFrame()
iteration_stats = pd.DataFrame(columns=["Iterations", "Mean_DSn", "SD_DSn"])

def run_simulation(n):
    DSn_list = []
    VQtz_list, VKfs_list, VAb_list, VBt_list, VMs_list, VIlm_list = ([] for _ in range(6))
    
    for _ in range(n):
        # Generate random mineral proportions and normalize them
        proportions = np.random.rand(6)
        proportions /= proportions.sum()

        VQtz, VKfs, VAb, VBt, VMs, VIlm = proportions
        
        # Geochemical constraints
        cond1 = VQtz*1 + VKfs*0.65 + VAb*0.69 + VBt*0.35 + VMs*0.45
        cond2 = VKfs*0.18 + VAb*0.19 + VBt*0.10 + VMs*0.38
        cond3 = VBt*0.42 + VIlm*0.47
        cond4 = VAb*0.12
        cond5 = VKfs*0.17 + VBt*0.09 + VMs*0.12
        cond6 = VIlm*0.53
        cond7 = VAb*0.12 + VKfs*0.17 + VBt*0.11 + VMs*0.12
        
        # Check whether all constraints are satisfied
        if (0.60 < cond1 < 0.80 and
            0.10 < cond2 < 0.20 and
            0 < cond3 < 0.05 and
            0 < cond4 < 0.07 and
            0 < cond5 < 0.07 and
            0 < cond6 < 0.001 and
            0 < cond7 < 0.10):

            # Calculate bulk Sn partition coefficient
            DSn = (VQtz*0 + VKfs*0.18 + VAb*0.0285 +
                   VBt*0.05 + VMs*0.53 + VIlm*0.32)
            DSn_list.append(DSn)
            
            VQtz_list.append(VQtz)
            VKfs_list.append(VKfs)
            VAb_list.append(VAb)
            VBt_list.append(VBt)
            VMs_list.append(VMs)
            VIlm_list.append(VIlm)

    return pd.DataFrame({
        "VQtz": VQtz_list, "VKfs": VKfs_list, "VAb": VAb_list,
        "VBt": VBt_list, "VMs": VMs_list, "VIlm": VIlm_list,
        "DSn": DSn_list
    })

for n in n_values:
    simulation_result = run_simulation(n)
    
    print("Iterations:", n)
    print("Number of valid combinations:", len(simulation_result))
    
    if len(simulation_result) > 0:
        simulation_result["Iterations"] = n
        valid_results_all = pd.concat([valid_results_all, simulation_result], ignore_index=True)
        
        mean_DSn = simulation_result["DSn"].mean()
        sd_DSn = 2 * simulation_result["DSn"].std()  # 2σ uncertainty
        
        iteration_stats.loc[len(iteration_stats)] = [n, mean_DSn, sd_DSn]
    else:
        iteration_stats.loc[len(iteration_stats)] = [n, np.nan, np.nan]

# Plot mean DSn versus iterations
plt.figure()
plt.plot(iteration_stats["Iterations"], iteration_stats["Mean_DSn"], marker="o")
plt.xscale("log")
plt.xlabel("Number of iterations")
plt.ylabel("Mean DSn")
plt.title("Evolution of mean DSn")
plt.show()

# Plot uncertainty of DSn versus iterations
plt.figure()
plt.plot(iteration_stats["Iterations"], iteration_stats["SD_DSn"], marker="o")
plt.xscale("log")
plt.xlabel("Number of iterations")
plt.ylabel("2σ uncertainty")
plt.title("Uncertainty of DSn with iterations")
plt.show()

print(iteration_stats)

# Save all valid solutions
valid_results_all.to_csv("D:/YS-Monte Carlo.csv", index=False)
