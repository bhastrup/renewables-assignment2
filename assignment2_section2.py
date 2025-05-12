import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Parameters
# -------------------------
np.random.seed(100)
total_profiles = 300
num_minutes = 60
min_load = 220
max_load = 600
max_delta = 35
M = 1e4
epsilon = 0.10

# -------------------------
# Pofiles generation
# -------------------------
def generate_profile(start=None):
    if start is None:
        start = np.random.uniform(min_load, max_load) 
    profile = [start]
    for i in range(1, num_minutes):
        prev = profile[-1]
        lb = max(min_load, prev - max_delta)
        ub = min(max_load, prev + max_delta)
        profile.append(np.random.uniform(lb, ub))
    return profile

#profile generation
profiles = np.array([generate_profile() for i in range(total_profiles)])

#in-sample and out-of-sample
in_sample_profiles = profiles[:100]
out_of_sample_profiles = profiles[100:]

#find F_up
F_up = in_sample_profiles - min_load
samples = list(np.ndindex(100, 60))

# -------------------------
# ALSO-X algorithm
# -------------------------
def ALSOX_milp(F_up_matrix, M, epsilon):
    model = gp.Model("ALSOX_MILP")
    model.Params.OutputFlag = 0
    model.Params.Method = 1
    q = int(epsilon * F_up_matrix.size)

    c_up = model.addVar(lb=0, name="c_up")
    y = model.addVars(len(samples), vtype=GRB.BINARY, name="y")

    for idx, (w, m) in enumerate(samples):
        model.addConstr(c_up - F_up_matrix[w, m] <= y[idx] * M)

    model.addConstr(gp.quicksum(y[idx] for idx in range(len(samples))) <= q)
    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"ALSO-X MILP optimal reserve bid: {model.ObjVal:.2f} kW")
        return model.ObjVal
    else:
        print("ALSO-X MILP did not find any solution")
        return None

# -------------------------
# Cvar
# -------------------------
def solve_cvar_lp(F_up_matrix, epsilon):
    model = gp.Model("CVaR_LP")
    model.Params.OutputFlag = 0
    model.Params.Method = 1
    total_samples = F_up_matrix.size

    c_up = model.addVar(lb=0, name="c_up")
    beta = model.addVar(ub=0, name="beta")
    zeta = model.addVars(len(samples), lb=0, name="zeta")

    for idx, (w, m) in enumerate(samples):
        model.addConstr(c_up - F_up_matrix[w, m] <= zeta[idx])
        model.addConstr(beta <= zeta[idx])

    model.addConstr(
        (1 / total_samples) * gp.quicksum(zeta[idx] for idx in range(len(samples))) <= (1 - epsilon) * beta,
        name="cvar_constraint"
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Cvar optimal reserve bid: {model.ObjVal:.2f} kW")
        return model.ObjVal
    else:
        print("Cvar did not find a solution")
        return None

# -------------------------
# Running the techniques
# -------------------------
also_x_result = ALSOX_milp(F_up, M, epsilon)
cvar_result = solve_cvar_lp(F_up, epsilon)

# -------------------------
# Out-of-sample analysis
# -------------------------
F_up_out = out_of_sample_profiles - min_load
total_samples_out = F_up_out.shape[0] * F_up_out.shape[1]

errors_also_x = np.sum(F_up_out < also_x_result)
errors_cvar = np.sum(F_up_out < cvar_result)

error_rate_also_x = errors_also_x / total_samples_out
error_rate_cvar = errors_cvar / total_samples_out


print(f"\nOut-of-sample analysis:")
print(f"ALSO-X error rate = {error_rate_also_x:.2%}")
print(f"CVaR error rate = {error_rate_cvar:.2%}")


# -------------------------
# Energinet perspective
# -------------------------
epsilons = np.arange(0.00, 0.21, 0.02)[::-1] 
F_up_out = out_of_sample_profiles - min_load
total_samples_out = F_up_out.shape[0] * F_up_out.shape[1]

results = []

for eps in epsilons:
    bid = ALSOX_milp(F_up, M, eps)
    errors = np.sum(F_up_out < bid)
    error_rate = errors / total_samples_out
    results.append((1 - eps, bid, error_rate))  

df = pd.DataFrame(results, columns=["P_level", "Reserve_bid_kW", "Out_of_sample_error_rate"])

#plotting 
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel("P")
ax1.set_ylabel("Optimal reserve bid [kW]", color=color)
ax1.plot(df["P_level"], df["Reserve_bid_kW"], marker='o', color=color, label="Reserve bid")
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_xaxis()

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Expected reserve shortfall", color=color)
ax2.plot(df["P_level"], df["Out_of_sample_error_rate"], marker='x', linestyle='--', color=color, label="Error rate")
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig("energinet_perspective.png", dpi=300, bbox_inches='tight')
plt.show()
print(df)
