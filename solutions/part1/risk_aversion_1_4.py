import os
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from prep_data import (
    get_wind_forecast,
    get_price_forecast,
    get_system_condition,
    combine_scenarios
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def split_scenarios(scenarios, n_is) -> tuple[dict, dict]:
    idx_is = np.random.choice(list(scenarios.keys()), size=n_is, replace=False)
    idx_oos = np.setdiff1d(list(scenarios.keys()), idx_is)
    scenarios_is = {idx: scenarios[idx] for idx in idx_is}
    scenarios_oos = {idx: scenarios[idx] for idx in idx_oos}
    return scenarios_is, scenarios_oos

def get_var_as_array(scenarios: dict, var_name: str) -> np.ndarray:
    return np.array([scenarios[k][var_name] for k in scenarios.keys()])

@dataclass
class ModelData:
    n_scenarios: int
    price: np.ndarray
    wind: np.ndarray
    c: np.ndarray
    w: float
    T: int
    balancing_coef_high: float
    balancing_coef_low: float
    max_capacity: float

class StochasticOfferingModel:
    def __init__(self, scenarios: dict, max_capacity: float,
                 beta: float = 0.0, alpha: float = 0.9,
                 balancing_coef_high: float = 1.25, balancing_coef_low: float = 0.85):
        self.beta = beta
        self.alpha = alpha
        d = self.prepare_data(scenarios, max_capacity, balancing_coef_high, balancing_coef_low)
        self.data = d  # Save data for external use

    @staticmethod
    def prepare_data(scenarios: dict, max_capacity: float, balancing_coef_high: float, balancing_coef_low: float):
        return ModelData(
            n_scenarios=len(scenarios),
            price=get_var_as_array(scenarios, "price_forecast"),
            wind=get_var_as_array(scenarios, "wind_forecast"),
            c=get_var_as_array(scenarios, "system_condition"),
            w=1 / len(scenarios),
            T=24,
            balancing_coef_high=balancing_coef_high,
            balancing_coef_low=balancing_coef_low,
            max_capacity=max_capacity
        )

class StochasticOfferingModelOnePrice(StochasticOfferingModel):
    def setup_gurobi_model(self, d: ModelData):
        model = gp.Model("One-Price Stochastic Offering with CVaR")

        p_da = model.addVars(d.T, lb=0, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="p_da")
        zeta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta")
        eta = model.addVars(d.n_scenarios, lb=0, vtype=GRB.CONTINUOUS, name="eta")

        balancing_price = d.price * (d.c * d.balancing_coef_low + (1 - d.c) * d.balancing_coef_high)

        profit_expr = {}
        for omega in range(d.n_scenarios):
            profit = gp.quicksum(
                d.price[omega, t] * p_da[t] + balancing_price[omega, t] * (d.wind[omega, t] - p_da[t])
                for t in range(d.T)
            )
            profit_expr[omega] = profit
            model.addConstr(eta[omega] >= zeta - profit, name=f"cvar_eta_{omega}")

        expected_profit = gp.quicksum(d.w * profit_expr[omega] for omega in range(d.n_scenarios))
        cvar_term = zeta - (1 / (1 - self.alpha)) * gp.quicksum(d.w * eta[omega] for omega in range(d.n_scenarios))
        model.setObjective(expected_profit + self.beta * cvar_term, GRB.MAXIMIZE)

        self.vars = {"p_da": p_da, "zeta": zeta, "eta": eta}
        self.data = d
        self.model = model
        self.profit_expr = profit_expr
        self.balancing_price = balancing_price

    def optimize(self):
        self.setup_gurobi_model(self.data)
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"
        return self

    def get_outcome(self):
        d = self.data
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])
        eta_vals = np.array([self.vars["eta"][omega].X for omega in range(d.n_scenarios)])

        day_ahead_profit = p_da * d.price  
        balancing_profit = self.balancing_price * (d.wind - p_da)

        expected_profit = np.mean(np.sum(day_ahead_profit + balancing_profit, axis=1))

        return {
            "p_da": p_da,
            "expected_profit": expected_profit,
            "zeta": self.vars["zeta"].X,
            "eta": eta_vals
        }

    def get_scenario_profits(self):
        d = self.data
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])
        day_ahead_profit = np.sum(p_da * d.price, axis=1)
        balancing_profit = np.sum(self.balancing_price * (d.wind - p_da), axis=1)
        return day_ahead_profit + balancing_profit

class StochasticOfferingModelTwoPrices(StochasticOfferingModel):
    def setup_gurobi_model(self, d: ModelData):
        model = gp.Model("Two-Price Stochastic Offering with CVaR")

        p_da = model.addVars(d.T, lb=0, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="p_da")
        delta = model.addVars(d.n_scenarios, d.T, lb=-d.max_capacity, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="delta")
        delta_up = model.addVars(d.n_scenarios, d.T, lb=0, vtype=GRB.CONTINUOUS, name="delta_up")
        delta_down = model.addVars(d.n_scenarios, d.T, lb=0, vtype=GRB.CONTINUOUS, name="delta_down")
        y = model.addVars(d.n_scenarios, d.T, vtype=GRB.BINARY, name="y")
        zeta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta")
        eta = model.addVars(d.n_scenarios, lb=0, vtype=GRB.CONTINUOUS, name="eta")

        model.addConstrs((delta[omega, t] == d.wind[omega, t] - p_da[t] for omega in range(d.n_scenarios) for t in range(d.T)))
        model.addConstrs((delta_up[omega, t] - delta_down[omega, t] == delta[omega, t] for omega in range(d.n_scenarios) for t in range(d.T)))
        model.addConstrs((delta_up[omega, t] <= d.max_capacity * y[omega, t] for omega in range(d.n_scenarios) for t in range(d.T)))
        model.addConstrs((delta_down[omega, t] <= d.max_capacity * (1 - y[omega, t]) for omega in range(d.n_scenarios) for t in range(d.T)))

        profit_expr = {}
        for omega in range(d.n_scenarios):
            profit = gp.quicksum(
                d.price[omega, t] * p_da[t] +
                d.price[omega, t] * (
                    delta_up[omega, t] * d.c[omega, t] * d.balancing_coef_low +
                    delta_up[omega, t] * (1 - d.c[omega, t]) -
                    delta_down[omega, t] * d.c[omega, t] -
                    delta_down[omega, t] * (1 - d.c[omega, t]) * d.balancing_coef_high
                )
                for t in range(d.T)
            )
            profit_expr[omega] = profit
            model.addConstr(eta[omega] >= zeta - profit, name=f"cvar_eta_{omega}")

        expected_profit = gp.quicksum(d.w * profit_expr[omega] for omega in range(d.n_scenarios))
        cvar_term = zeta - (1 / (1 - self.alpha)) * gp.quicksum(d.w * eta[omega] for omega in range(d.n_scenarios))
        model.setObjective(expected_profit + self.beta * cvar_term, GRB.MAXIMIZE)

        self.vars = {
            "p_da": p_da,
            "delta": delta,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "y": y,
            "zeta": zeta,
            "eta": eta
        }
        self.data = d
        self.model = model
        self.profit_expr = profit_expr

    def optimize(self):
        self.setup_gurobi_model(self.data)
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"
        return self

    def get_outcome(self):
        d = self.data
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])
        eta_vals = np.array([self.vars["eta"][omega].X for omega in range(d.n_scenarios)])

        delta_up = np.array([self.vars["delta_up"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)
        delta_down = np.array([self.vars["delta_down"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)

        price = d.price
        c = d.c

        day_ahead_profit = p_da * price  
        balancing_profit = price * (
            delta_up * c * d.balancing_coef_low +
            delta_up * (1 - c) -
            delta_down * c -
            delta_down * (1 - c) * d.balancing_coef_high
        )
        expected_profit = np.mean(np.sum(day_ahead_profit + balancing_profit, axis=1))  

        return {
            "p_da": p_da,
            "expected_profit": expected_profit,
            "zeta": self.vars["zeta"].X,
            "eta": eta_vals
        }

    def get_scenario_profits(self):
        d = self.data
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])
        delta_up = np.array([self.vars["delta_up"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)
        delta_down = np.array([self.vars["delta_down"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)

        price = d.price
        c = d.c

        day_ahead_profit = np.sum(p_da * price, axis=1)
        balancing_profit = np.sum(price * (
            delta_up * c * d.balancing_coef_low +
            delta_up * (1 - c) -
            delta_down * c -
            delta_down * (1 - c) * d.balancing_coef_high
        ), axis=1)

        return day_ahead_profit + balancing_profit
    
if __name__ == "__main__":
    set_seed(42)

    dataset_dir = "datasets/"
    plot_dir = "results/Part1/"
    os.makedirs(plot_dir, exist_ok=True)

    wind = get_wind_forecast(dataset_dir, n_scenarios=20)
    price = get_price_forecast(dataset_dir, n_scenarios=20)
    cond = get_system_condition(p=0.5, n_scenarios=4)
    scenarios = combine_scenarios(wind, price, cond)
    scenarios_is, _ = split_scenarios(scenarios, n_is=200)

    beta_values = np.linspace(0, 1.5, 16)
    alpha = 0.9

    results_one = {"beta": [], "profit": [], "cvar": []}
    results_two = {"beta": [], "profit": [], "cvar": []}

    for beta in beta_values:
        model_one = StochasticOfferingModelOnePrice(scenarios_is, wind.max(), beta, alpha).optimize()
        res_one = model_one.get_outcome()
        profit_one = res_one["expected_profit"]
        cvar_one = res_one["zeta"] - (1 / (1 - alpha)) * np.sum(model_one.data.w * res_one["eta"])
        results_one["beta"].append(beta)
        results_one["profit"].append(profit_one)
        results_one["cvar"].append(cvar_one)

        model_two = StochasticOfferingModelTwoPrices(scenarios_is, wind.max(), beta, alpha).optimize()
        res_two = model_two.get_outcome()
        profit_two = res_two["expected_profit"]
        cvar_two = res_two["zeta"] - (1 / (1 - alpha)) * np.sum(model_two.data.w * res_two["eta"])
        results_two["beta"].append(beta)
        results_two["profit"].append(profit_two)
        results_two["cvar"].append(cvar_two)

    beta_vals = [0.0, 0.5, 1.0, 1.5]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for beta in beta_vals:
        scenarios_is, _ = split_scenarios(scenarios, n_is=100)

        model_one = StochasticOfferingModelOnePrice(scenarios_is, wind.max(), beta, alpha).optimize()
        profits_one = model_one.get_scenario_profits()
        axs[0].hist(profits_one, bins=20, alpha=0.6, label=f"beta={beta:.1f}")

        model_two = StochasticOfferingModelTwoPrices(scenarios_is, wind.max(), beta, alpha).optimize()
        profits_two = model_two.get_scenario_profits()
        axs[1].hist(profits_two, bins=20, alpha=0.6, label=f"beta={beta:.1f}")

    axs[0].set_title("One price")
    axs[1].set_title("Two price")
    for ax in axs:
        ax.set_xlabel("Profit [EUR]")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "profit_distribution_comparison.png"))
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(results_one["cvar"], results_one["profit"], marker="o", color="blue")
    axs[0].set_title("One price")
    axs[1].plot(results_two["cvar"], results_two["profit"], marker="s", color="red")
    axs[1].set_title("Two price")

    for ax in axs:
        ax.set_xlabel("CVaR [EUR]")
        ax.set_ylabel("Expected profit [EUR]")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "expected_profit_vs_cvar_side_by_side.png"))
    plt.show()

    scenario_counts = [50, 100, 200]
    seeds = [1, 2, 3]
    beta = 1.0
    alpha = 0.9
    
    results_size_one = {"size": [], "cvar": [], "profit": []}
    results_size_two = {"size": [], "cvar": [], "profit": []}
    
    for n in scenario_counts:
        for seed in seeds:
            set_seed(seed)
            scenarios_is, _ = split_scenarios(scenarios, n_is=n)
    
            model_one = StochasticOfferingModelOnePrice(scenarios_is, wind.max(), beta, alpha).optimize()
            res_one = model_one.get_outcome()
            cvar_one = res_one["zeta"] - (1 / (1 - alpha)) * np.sum(model_one.data.w * res_one["eta"])
            profit_one = res_one["expected_profit"]
            results_size_one["size"].append(n)
            results_size_one["cvar"].append(cvar_one)
            results_size_one["profit"].append(profit_one)
    
            model_two = StochasticOfferingModelTwoPrices(scenarios_is, wind.max(), beta, alpha).optimize()
            res_two = model_two.get_outcome()
            cvar_two = res_two["zeta"] - (1 / (1 - alpha)) * np.sum(model_two.data.w * res_two["eta"])
            profit_two = res_two["expected_profit"]
            results_size_two["size"].append(n)
            results_size_two["cvar"].append(cvar_two)
            results_size_two["profit"].append(profit_two)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    colors = {50: "blue", 100: "orange", 200: "green"}
    
    for i in range(len(results_size_one["size"])):
        n = results_size_one["size"][i]
        axs[0].scatter(results_size_one["cvar"][i], results_size_one["profit"][i], color=colors[n], label=f"{n} scenarios" if i % 3 == 0 else "")
    for i in range(len(results_size_two["size"])):
        n = results_size_two["size"][i]
        axs[1].scatter(results_size_two["cvar"][i], results_size_two["profit"][i], color=colors[n], label=f"{n} scenarios" if i % 3 == 0 else "")
    
    axs[0].set_title("One price")
    axs[0].set_xlabel("CVaR [EUR]")
    axs[0].set_ylabel("Expected profit [EUR]")
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_title("Two price")
    axs[1].set_xlabel("CVaR [EUR]")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sensitivity_to_scenario_count_multi_seed.png"))
    plt.show()
