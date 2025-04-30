import os, glob
from copy import deepcopy, copy
from dataclasses import dataclass
import random
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from matplotlib import pyplot as plt

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
    """ Split scenarios into in-sample and out-of-sample sets. """
    idx_is = np.random.choice(list(scenarios.keys()), size=n_is, replace=False) # np.ndarray of shape (n_is,)
    idx_oos = np.setdiff1d(list(scenarios.keys()), idx_is) # np.ndarray of shape (n_oos,)

    scenarios_is = {idx: scenarios[idx] for idx in idx_is}
    scenarios_oos = {idx: scenarios[idx] for idx in idx_oos}

    return scenarios_is, scenarios_oos


def get_var_as_array(scenarios: dict, var_name: str) -> np.ndarray:
    return np.array([scenarios[k][var_name] for k in scenarios.keys()])



@dataclass
class ModelData:
    n_scenarios: int
    price: np.ndarray # (n_scenarios, T)
    wind: np.ndarray # (n_scenarios, T)
    c: np.ndarray # (n_scenarios, T)
    w: float 
    T: int 
    balancing_coef_high: float
    balancing_coef_low: float
    max_capacity: float



class StochasticOfferingModel:
    def __init__(
            self, 
            scenarios: dict,
            max_capacity: float,
            balancing_coef_high: float = 1.25,
            balancing_coef_low: float = 0.85
        ):
        d = self.prepare_data(scenarios, max_capacity, balancing_coef_high, balancing_coef_low)
        self.setup_gurobi_model(d)

    def setup_gurobi_model(self, d: ModelData):

        model = gp.Model("One-Price Stochastic Offering")


        # Calculate balancing price
        self.balancing_price = d.price * (d.c * d.balancing_coef_low + (1 - d.c) * d.balancing_coef_high)

 
        # Decision variables: Day-ahead production
        p_da = model.addVars(d.T, lb=0, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="p_da")

        # Objective function: Day-ahead profit + balancing profit
        model.setObjective(
            gp.quicksum(
                d.w * (
                    d.price[omega, t] * p_da[t] +                                       # Day-ahead profit
                    self.balancing_price[omega, t] * (d.wind[omega, t] - p_da[t])       # Balancing profit
                )
                for omega in range(d.n_scenarios) for t in range(d.T)
            ),
            GRB.MAXIMIZE,
        )
        
        self.vars = {
            "p_da": p_da,
        }
        self.data = d
        self.model = model

    def optimize(self):
        self.model.optimize()
        assert self.model.status == GRB.OPTIMAL, "Model is not optimal"
        return self
    
    @staticmethod
    def prepare_data(scenarios: dict, max_capacity: float, balancing_coef_high: float, balancing_coef_low: float):
        return ModelData(
            n_scenarios = len(scenarios),
            price = get_var_as_array(scenarios, "price_forecast"),
            wind = get_var_as_array(scenarios, "wind_forecast"),
            c = get_var_as_array(scenarios, "system_condition"),
            w = 1 / len(scenarios),
            T = 24,
            balancing_coef_high = balancing_coef_high,
            balancing_coef_low = balancing_coef_low,
            max_capacity = max_capacity
        )

    def get_outcome(self, oos_scenarios: Optional[dict] = None):
        d = self.data
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])

        # Calculate day-ahead profit. Always use in-sample scenarios
        day_ahead_profit_array = p_da * d.price # (n_scenarios, 24)

        # Balancing profit
        if oos_scenarios is None:
            balancing_profit_array = self.balancing_price * (d.wind - p_da) # (n_scenarios, 24)
        else:
            d_oos = self.prepare_data(oos_scenarios, d.max_capacity, d.balancing_coef_high, d.balancing_coef_low)
            balancing_price_oos = d_oos.price * (d_oos.c * d_oos.balancing_coef_low + (1 - d_oos.c) * d_oos.balancing_coef_high)
            balancing_profit_array = balancing_price_oos * (d_oos.wind - p_da) # (n_scenarios, 24)

        # Average over scenarios
        expected_day_ahead_profit = np.mean(day_ahead_profit_array, axis=0) # (24,)
        expected_balancing_profit = np.mean(balancing_profit_array, axis=0) # (24,)
        expected_profit = expected_day_ahead_profit + expected_balancing_profit # (24,)

        if oos_scenarios is None:
            assert expected_profit.sum() - self.model.getObjective().getValue() < 1e-2, \
                f"{expected_profit.sum()} != {self.model.getObjective().getValue()}"

        return {
            "p_da": p_da,
            "obj": self.model.getObjective().getValue(),
            "day_ahead_profit": expected_day_ahead_profit,
            "balancing_profit": expected_balancing_profit,
            "expected_profit": expected_profit
        }



class StochasticOfferingModelTwoPrices(StochasticOfferingModel):
    def setup_gurobi_model(self, d: ModelData):

        model = gp.Model("Two-Price Stochastic Offering")
    
        # Decision variables: Day-ahead production
        p_da = model.addVars(d.T, lb=0, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="p_da")

        # Auxiliary variables
        delta = model.addVars(range(d.n_scenarios), range(d.T), lb=-d.max_capacity, ub=d.max_capacity, vtype=GRB.CONTINUOUS, name="delta")
        delta_up = model.addVars(range(d.n_scenarios), range(d.T), lb=0, vtype=GRB.CONTINUOUS, name="delta_up")
        delta_down = model.addVars(range(d.n_scenarios), range(d.T), lb=0, vtype=GRB.CONTINUOUS, name="delta_down")
        y = model.addVars(range(d.n_scenarios), range(d.T), vtype=GRB.BINARY, name="y")


        # Objective function: Day-ahead profit + balancing profit where we overwrite favorable prices
        model.setObjective(
            gp.quicksum(
                d.w * (
                    d.price[omega, t] * p_da[t] +                                               # Day-ahead profit
                    d.price[omega, t] * (
                        delta_up[omega, t] * d.c[omega, t] * d.balancing_coef_low +             # Producer in excess, system in excess
                        delta_up[omega, t] * (1 - d.c[omega, t]) * 1                            # Producer in excess, system in deficit
                        - delta_down[omega, t] * d.c[omega, t] * 1                              # Producer in deficit, system in excess
                        - delta_down[omega, t] * (1 - d.c[omega, t]) * d.balancing_coef_high    # Producer in deficit, system in deficit
                    )
                )
                for omega in range(d.n_scenarios) for t in range(d.T)
            ),
            GRB.MAXIMIZE,
        )

        ######### Constraints #########
        # Producer imbalance
        model.addConstrs(
            delta[omega, t] == d.wind[omega, t] - p_da[t] \
                for omega in range(d.n_scenarios) for t in range(d.T)
        )
        # Relate delta to unsigned deltas (delta_up and delta_down)
        model.addConstrs(
            delta_up[omega, t] - delta_down[omega, t] == delta[omega, t] \
                for omega in range(d.n_scenarios) for t in range(d.T)
        )
        # Force product of delta_up and delta_down to be 0
        model.addConstrs(
            delta_up[omega, t] <= d.max_capacity * y[omega, t]
            for omega in range(d.n_scenarios) for t in range(d.T)
        )
        model.addConstrs(
            delta_down[omega, t] <= d.max_capacity * (1 - y[omega, t])
            for omega in range(d.n_scenarios) for t in range(d.T)
        )

        self.vars = {
            "p_da": p_da,
            "delta": delta,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "y": y
        }
        self.data = d
        self.model = model


    @staticmethod
    def calc_balancing_profit(
            price: np.ndarray,
            delta_up: np.ndarray,
            delta_down: np.ndarray,
            c: np.ndarray,
            balancing_coef_high: float,
            balancing_coef_low: float,
    ):

        return price * (
            delta_up * c * balancing_coef_low +                 # Producer in excess, system in excess
            delta_up * (1 - c) * 1 -                            # Producer in excess, system in deficit
            delta_down * c * 1 -                                # Producer in deficit, system in excess
            delta_down * (1 - c) * balancing_coef_high          # Producer in deficit, system in deficit
        )


    def get_outcome(self, oos_scenarios: Optional[dict] = None):
        d = self.data

        # Get the values of the variables from the model solution
        p_da = np.array([self.vars["p_da"][i].X for i in range(d.T)])
        delta = np.array([self.vars["delta"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)
        delta_up = np.array([self.vars["delta_up"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)
        delta_down = np.array([self.vars["delta_down"][(i, j)].X for i in range(d.n_scenarios) for j in range(d.T)]).reshape(d.n_scenarios, d.T)
        assert np.all(delta_up*delta_down == 0)


        # Calculate day-ahead profit. Always use in-sample scenarios
        day_ahead_profit_array = p_da * d.price # (n_scenarios, 24)


        if oos_scenarios is None:
            # Balancing profit
            balancing_profit_array = self.calc_balancing_profit(
                price=d.price,
                delta_up=delta_up,
                delta_down=delta_down,
                c=d.c,
                balancing_coef_high=d.balancing_coef_high,
                balancing_coef_low=d.balancing_coef_low
            )
        else:

            d_oos = self.prepare_data(oos_scenarios, d.max_capacity, d.balancing_coef_high, d.balancing_coef_low)
            delta_oos = d_oos.wind - p_da
            delta_up_oos = np.maximum(delta_oos, 0)
            delta_down_oos = np.abs(np.minimum(delta_oos, 0))

            # Balancing profit Out-of-sample
            balancing_profit_array = self.calc_balancing_profit(
                price=d_oos.price,
                delta_up=delta_up_oos,
                delta_down=delta_down_oos,
                c=d_oos.c,
                balancing_coef_high=d_oos.balancing_coef_high,
                balancing_coef_low=d_oos.balancing_coef_low
            )


        # Average over scenarios
        expected_day_ahead_profit = np.mean(day_ahead_profit_array, axis=0) # (24,)
        expected_balancing_profit = np.mean(balancing_profit_array, axis=0) # (24,)
        expected_profit = expected_day_ahead_profit + expected_balancing_profit # (24,)
        
        if oos_scenarios is None:
            assert expected_profit.sum() - self.model.getObjective().getValue() < 1e-2, \
                f"{expected_profit.sum()} != {self.model.getObjective().getValue()}"

        return {
            "p_da": p_da,
            "obj": self.model.getObjective().getValue(),
            "day_ahead_profit": expected_day_ahead_profit,
            "balancing_profit": expected_balancing_profit,
            "expected_profit": expected_profit
        }


def print_results_1_1_and_1_2(outcome: dict, scenarios_is: np.ndarray, plot_dir: str, part: str):    

    print(f"Optimal hourly production quantity offers of the wind farm in the day-ahead market:")
    for t, p in zip(range(24), outcome["p_da"]):
        print(f"Hour {t}-{t+1}: {p:.2f} MWh")

    print(f"Expected hourly profit:")
    for t, p in zip(range(24), outcome["expected_profit"]):
        print(f"Hour {t}-{t+1}: {p:.2f} EUR")

    total_profit = outcome["expected_profit"].sum()
    print(f"Expected total profit: {total_profit:.2f} EUR")

    cumulative_profit = np.cumsum(outcome["expected_profit"])
    print(f"Expected cumulative profit:")
    for t, p in zip(range(24), cumulative_profit):
        print(f"Hour {t}-{t+1}: {p:.2f} EUR")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Top subplot - profits
    x = np.arange(1, 25)
    x = np.insert(x, 0, 0)
    y = np.insert(cumulative_profit, 0, 0)
    ax1.plot(x, y, '*--', label="Cumulative profit")
    ax1.bar(0.5 + np.arange(24), outcome["expected_profit"], color="blue", label="Hourly profit")
    ax1.legend()
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Profit (EUR)")
    ax1.set_xlim(0, 24)
    ax1.grid(True)
    ax1.text(12, 0.95*total_profit, f"Total profit: {total_profit/1e6:.2f}M EUR", fontsize=12)

    # Bottom subplot - quantity offers
    ax2.bar(0.5 + np.arange(24), outcome["p_da"], color="green", label="Day-ahead offers")

    # Plot the wind forecast for the in-sample scenarios
    wind_forecast = get_var_as_array(scenarios_is, "wind_forecast")
    for i in range(wind_forecast.shape[0]):
        ax2.plot(0.5 + np.arange(24), wind_forecast[i, :], '-', color="red", label="Physical production scenarios" if i == 0 else None, linewidth=0.1)
    
    ax2.plot(0.5 + np.arange(24), wind_forecast.mean(axis=0), '-', color="blue", label="Expected physical production")
    
    # Get legend handles and labels
    handles, labels = ax2.get_legend_handles_labels()
    
    # Create new handles with thicker lines for better visibility
    handles = [copy(ha) for ha in handles]
    [ha.set_linewidth(5) for ha in handles if hasattr(ha, 'set_linewidth')]
    
    # Explicitly pass both handles and labels to legend
    ax2.legend(handles=handles, labels=labels, loc="upper right")
    ax2.set_xlabel("Hour") 
    ax2.set_ylabel("Quantity (MWh)")
    ax2.set_xlim(0, 24)
    ax2.grid(True)
    
    plt.tight_layout()

    plt.savefig(plot_dir + f"cumulative_profit_{part}.png")
    plt.close()


class KFoldIterator:
    def __init__(self, scenarios: dict, n_folds: int, shuffle: bool = True):
        assert len(scenarios) % n_folds == 0, "Number of scenarios must be divisible by number of folds"

        self.scenarios = deepcopy(scenarios)
        self.n_folds = n_folds
        self.N = len(scenarios)
        self.n_is = len(scenarios) // n_folds

        keys = list(scenarios.keys())

        if shuffle:
            np.random.shuffle(keys)

        # Create folds
        self.folds = []
        for k in range(n_folds):
            is_idx = np.arange(k*self.n_is, (k+1)*self.n_is)
            oos_idx = np.setdiff1d(np.arange(self.N), is_idx)

            fold_is = {keys[i]: self.scenarios[keys[i]] for i in is_idx}
            fold_oos = {keys[i]: self.scenarios[keys[i]] for i in oos_idx}
            self.folds.append((fold_is, fold_oos))


    def __iter__(self):
        self.current_fold = 0
        return self

    def __next__(self):
        if self.current_fold < self.n_folds:
            fold = self.folds[self.current_fold]
            self.current_fold += 1
            return fold
        else:
            raise StopIteration


def plot_profits_across_folds(profits_is, profits_oos, model_name: str, n_folds: int):
    x = np.arange(24)+0.5
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # set overall title
    fig.suptitle(f"{model_name}, {n_folds} folds", fontsize=16)

    # Hourly profits plot
    for i in range(n_folds):
        ax1.plot(x, profits_is[i], label=None, c="blue", linewidth=0.5)
        ax1.plot(x, profits_oos[i], label=None, c="red", linewidth=0.5)
    ax1.plot(x, profits_is.mean(axis=0), label=f"In-sample", c="blue", linewidth=3)
    ax1.plot(x, profits_oos.mean(axis=0), label=f"Out-of-sample", c="red", linewidth=3)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Profit (EUR)")
    ax1.legend()
    ax1.set_title("Hourly Profits")

    # Cumulative profits plot
    x = np.arange(0, 25)
    cum_profits_is = np.cumsum(np.hstack([np.zeros((n_folds, 1)), profits_is]), axis=1) # (n_folds, 24)
    cum_profits_oos = np.cumsum(np.hstack([np.zeros((n_folds, 1)), profits_oos]), axis=1) # (n_folds, 24)
    for i in range(n_folds):
        ax2.plot(x, cum_profits_is[i], label=None, c="blue", linewidth=0.5)
        ax2.plot(x, cum_profits_oos[i], label=None, c="red", linewidth=0.5)
    ax2.plot(x, cum_profits_is.mean(axis=0), label=f"In-sample", c="blue", linewidth=3)
    ax2.plot(x, cum_profits_oos.mean(axis=0), label=f"Out-of-sample", c="red", linewidth=3)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Cumulative Profit (EUR)")
    ax2.legend()
    ax2.set_title("Cumulative Profits")

    # Total expected profits boxplot
    total_profits = np.sum(np.stack([profits_is, profits_oos]), axis=2)  # (2, n_folds)
    ax3.boxplot(total_profits.T, tick_labels=['In-sample', 'Out-of-sample'])
    ax3.set_ylabel("Total Profit (EUR)")
    ax3.set_title("Total Expected Profits")


    plt.tight_layout(w_pad=2.0)
    plt.savefig(plot_dir + f"ex_post_{model_name}_{n_folds}_folds.png")
    plt.close()



if __name__ == "__main__":
    set_seed(42)

    dataset_dir = "datasets/"
    plot_dir = "results/Part1/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    
    # Parameters
    n_wind_forecasts = 20
    n_price_forecasts = 20
    n_system_conditions = 4
    n_is = 200


    # Load and process the data
    wind_forecast = get_wind_forecast(dataset_dir, n_scenarios=n_wind_forecasts) # (n_wind_forecasts, 24)
    price_forecast = get_price_forecast(dataset_dir, n_scenarios=n_price_forecasts) # (n_price_forecasts, 24)
    system_condition = get_system_condition(p=0.5, n_scenarios=n_system_conditions) # (n_system_conditions, 24)
    n_scenarios = n_wind_forecasts * n_price_forecasts * n_system_conditions
    n_oos = n_scenarios - n_is

    scenarios = combine_scenarios(wind_forecast, price_forecast, system_condition)
    scenarios_is, scenarios_oos = split_scenarios(scenarios, n_is)


    print("-"*100)
    print("1.1: Offering Strategy Under a One-Price Balancing Scheme")
    model = StochasticOfferingModel(
        scenarios=deepcopy(scenarios_is), 
        max_capacity=wind_forecast.max()
    ).optimize()
    outcome = model.get_outcome()
    print_results_1_1_and_1_2(outcome, scenarios_is, plot_dir, "1.1")

    
    print("-"*100)
    print("1.2: Offering Strategy Under a Two-Price Balancing Scheme")
    model = StochasticOfferingModelTwoPrices(
        scenarios=deepcopy(scenarios_is), 
        max_capacity=wind_forecast.max()
    ).optimize()
    outcome = model.get_outcome()
    print_results_1_1_and_1_2(outcome, scenarios_is, plot_dir, "1.2")



    print("-"*100)
    print("1.3: Ex-post Analysis")

    reduce_scenarios = False # Reduce the number of scenarios for testing
    if reduce_scenarios:
        scenarios, _ = split_scenarios(scenarios, n_is=160)
    
    def run_cv(scenarios: dict, model_cls, n_folds: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        profits_is, profits_oos = [], []
        for k, (fold_is, fold_oos) in enumerate(KFoldIterator(scenarios, n_folds=n_folds)):
            model = model_cls(deepcopy(fold_is), max_capacity=wind_forecast.max()).optimize()
            outcome_is = model.get_outcome()
            outcome_oos = model.get_outcome(fold_oos)
            profits_is.append(outcome_is["expected_profit"])
            profits_oos.append(outcome_oos["expected_profit"])
            print(f"[{name}] Fold {k+1}/{n_folds} | IS Profit: {outcome_is['expected_profit'].sum():.2f} | OOS Profit: {outcome_oos['expected_profit'].sum():.2f}")

        return np.array(profits_is), np.array(profits_oos)

    for n_folds in [4, 8, 16]:
        profits_1p_is, profits_1p_oos = run_cv(scenarios, StochasticOfferingModel, n_folds, "One-Price")
        profits_2p_is, profits_2p_oos = run_cv(scenarios, StochasticOfferingModelTwoPrices, n_folds, "Two-Price")
        plot_profits_across_folds(profits_1p_is, profits_1p_oos, model_name='One-Price', n_folds=n_folds)
        plot_profits_across_folds(profits_2p_is, profits_2p_oos, model_name='Two-Price', n_folds=n_folds)

