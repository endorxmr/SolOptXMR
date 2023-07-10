#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sunrise_lib

from ortools.algorithms import pywrapknapsack_solver


def solver(devices:dict, timeframe:int, power_limit:int, energy_limit:int|float) -> dict:
    """
    Calculate the mining solution for a given timeframe, power limit, and energy limit.

    Args:
        devices (dict): Unzipped dict of mining devices available.
        timeframe (int): Duration of the timeframe in minutes. [min]
        power_limit (int): Maximum power available for mining in Watts. [W]
        energy_limit (int|float): Energy available for mining in this timeframe in kilowatt-hours. [kWh]

    Returns:
        list(dict): List of mining devices that will be used.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.
        ValueError: If any of the input arguments have negative values.

    """

    # Type checking
    if not isinstance(timeframe, int):
        raise TypeError("The 'timeframe' argument must be of type 'int'.")
    if not isinstance(power_limit, int):
        raise TypeError("The 'power_limit' argument must be of type 'int'.")
    if not isinstance(energy_limit, int|float):
        raise TypeError("The 'energy_limit' argument must be of type 'int' or 'float'.")
    if not isinstance(devices, dict):
        raise TypeError("'devices' must be a dict")

    # Value checking
    if timeframe < 0 or power_limit < 0 or energy_limit < 0:
        raise ValueError("Input arguments cannot have negative values.")
    if len(devices) == 0:  # Empty list of devices, skipping computation and returning the same empty list
        return devices

    # Create the solver.
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    # TODO: figure out a way to ensure the minimum efficiency. An unconstrained solver might create a solution
    # that drops below the minimum profitability, if there's enough power and energy available. (Actually this
    # can even happen at the lower end).
    # Maybe this could come as a modified power/energy constraint? i.e. calculate the maximum power/energy that
    # we can consume at the current breakeven efficiency in our timeframe, and use that as the limit if it's a
    # tighter constraint than the current available power/energy.
    # Alternatively, we can take the unconstrained solution, and then remove items one by one (starting from
    # the least efficient ones) until the efficiency meets breakeven.

    # TODO: to avoid pointless calculations, do a quick sanity check: ensure that at least *one* of the available
    # devices has efficiency greater than breakeven. If there are none, avoid the calculation entirely (because
    # we would be mining at a loss no matter what).
    # Note: this needs a few specifiers:
    # - if we're disconnected from the grid, then any extra energy beyond what we can store (and beyond what the
    #   user sets as minimum charge level) would be wasted, so it's always profitable to mine - NOTE: this would
    #   already be reflected in the breakeven profitability being 0; this also applies if we're overcharging the
    #   batteries
    # - if we're connected to the grid and we sell back, then we also need to beat the energy buyback price, and
    #   not just basic profitability

    # TODO: don't forget about the difficulty prediction from tsqsim, which affects future profitability calculations!

    # TODO: add energy usage weights and capacity. Which could just be power usage * time frame duration.
    # We need both because the power limit is a limitation of the power delivery system (not fixed, depends on
    # system load), while the energy limit will change for each time frame (and will come as a function parameter).
    # ACTUALLY, the power available might also change for each timeframe, because we have to take into account
    # any other external consumers!

    # hashrates = [1000, 1000, 2920, 2920, 2920, 680*12, 680*12, 680*12, 680*12, 680*12]  # H
    # powers = [46, 46, 7.3*4, 7.3*4, 7.3*4, 73.8, 73.8, 73.8, 73.8, 73.8]  # W
    device_indices = devices["device_indices"]
    hashrates = devices["hashrates"]
    powers = devices["powers"]
    energy_const = timeframe / 60 / 1000  # [min] / 60 [min/h] / 1000 [W/kW]
    energies = [p * energy_const for p in powers]  # kWh
    # energies = powers  # W!!

    values = [round(x) for x in hashrates]  # H
    weights = [powers, energies]
    power_factor = 2  # Multiply all powers by 100 and round them off
    energy_factor = 7  # Multiply all energies by 1e7 and round them off
    weights[0] = [int(x * 10 ** power_factor) for x in weights[0]]  # W * power_factor
    weights[1] = [int(x * 10 ** energy_factor) for x in weights[1]]  # kWh * energy_factor
    capacities = [
        math.floor(power_limit * 10 ** power_factor),  # W * power_factor
        math.floor(energy_limit * 10 ** energy_factor)  # kWh * energy_factor
        # Instead of applying energy conversions to the weights, we could apply the inverse to the capacity
        # math.floor(energy_limit / energy_const * 10 ** energy_factor)  # W!! * energy_factor
        ]

    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()

    packed_items = []
    packed_weights = [[], []]
    total_weight = [0, 0]
    print('Capacities:', int(capacities[0] / 10**power_factor), capacities[1] / 10**energy_factor)
    # print('Capacities:', int(capacities[0] / 10**power_factor), capacities[1] * energy_const / 10**energy_factor)
    print('Total value =', computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights[0].append(weights[0][i] / 10**power_factor)
            packed_weights[1].append(weights[1][i] / 10**energy_factor)
            # packed_weights[1].append(weights[1][i] * energy_const / 10**energy_factor)
            total_weight[0] += weights[0][i] / 10**power_factor
            total_weight[1] += weights[1][i] / 10**energy_factor
            # total_weight[1] += weights[1][i] * energy_const / 10**energy_factor
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights[0], packed_weights[1])

    solution = {
        "chosen_indices": packed_items,
        "hashrate": computed_value,
        "power": total_weight[0],
        "energy": total_weight[1]
    }
    return solution


def setup_devices(devices:list[dict]) -> dict:
    if not isinstance(devices, list):
        raise TypeError("'devices' must be a list of dicts")
    if len(devices) == 0:  # Empty list of devices, returning the same empty list
        return devices

    device_indices = [j for j in range(len(devices)) for i in range(devices[j]["count"])]
    hashrates = [device["hash_per_core"] * device["cores"] for device in devices for _ in range(device["count"])]
    powers = [device["watt_per_core"] * device["cores"] for device in devices for _ in range(device["count"])]

    # devices = list(zip(device_indices, hashrates, powers, energies))  # List of tuples
    # keys = ["index", "hashrate", "power", "energy"]
    # devices = [{keys[i]: tup[i] for i in range(len(keys))} for tup in devices]  # List of dicts
    # devices = [device_indices, hashrates, powers, energies]  # List of unzipped lists
    # devices = {"device_indices": device_indices, "hashrates": hashrates, "powers": powers, "energies": energies}  # Dict of unzipped lists
    devices = {"device_indices": device_indices, "hashrates": hashrates, "powers": powers}  # Dict of unzipped lists
    return devices


if __name__ == '__main__':
    # Read the computers configuration
    computers = list(sunrise_lib.config_computers.get('computers'))
    devices = setup_devices(computers)
    timeframe = 5  # [min]
    power = 250  # [W]
    energy = 1  # [kWh]

    print(solver(devices, timeframe, power, energy))
