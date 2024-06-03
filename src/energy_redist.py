import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12.5, 7.5]
plt.rcParams['figure.dpi'] = 100
import numpy as np
from numpy.testing import assert_almost_equal
from enum import Enum

class Limitation(Enum):
    NO = 0
    POWER = 1
    MINER = 2
    ENERGY = 3

class EnergyBucket:
    def __init__(self, index, value):
        self.index = index  # Index of the bucket
        self.available = 0.0  # Initial available energy in the bucket
        self.filled = 0.0  # Initial filled energy in the bucket
        self.power_limited = False  # Initial power limitation status of the bucket
        self.miner_limited = False  # Initial miner limitation status of the bucket
        self.limitation = Limitation.NO  # Initial value of power/energy limitation status of the bucket
        self.value = value  # Value of the bucket


def calculate_battery_charge(interval: float, tf: np.ndarray, battery_params: dict, solar_input: np.ndarray, other_output: np.ndarray, mining_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the battery charge levels over the prediction timeframe

    ### Arguments:
        - interval {float}: [h] duration of each timeframe, in hours
        - tf {np.ndarray}: numpy array of the timeframes
        - battery_params {dict}: dict of battery parameters
        - solar_input {np.ndarray}: [W] numpy array of the solar power generated
        - other_output {np.ndarray}: [W] numpy array of the power used by non-mining devices
        - mining_output {np.ndarray}: [W] numpy array of the power used by mining equipment
    
    ### Raises:
        - ValueError: if the battery charge drops below 0 Wh at any point
    
    Returns:
        - battery_charge {np.ndarray}: [kWh] numpy array of the battery charge levels
        - excess_energy {np.ndarray}: [kWh] numpy array of the generated energy that exceeds the battery capacity
        - above_threshold {np.ndarray}: [kWh] numpy array of the battery energy above the usable threshold
    """
    initial_battery_charge = battery_params['initial_battery_charge']
    threshold = battery_params['threshold']
    max_battery_charge_kwh = battery_params['max_battery_charge_kwh']
    battery_charge = np.zeros_like(tf, dtype=float)
    battery_charge[0] = initial_battery_charge  # Initial value of battery charge in kWh
    battery_charge_hyp = np.zeros_like(tf, dtype=float)
    battery_charge_hyp[0] = initial_battery_charge  # Initial value of battery charge in kWh
    energy_added = np.zeros_like(tf, dtype=float)
    energy_consumed = np.zeros_like(tf, dtype=float)
    excess_energy = np.zeros_like(tf, dtype=float)
    above_threshold = np.zeros_like(tf, dtype=float)
    above_threshold[0] = battery_charge[0] - threshold if battery_charge[0] > threshold else 0
    # above_threshold_hyp = np.zeros_like(tf, dtype=float)
    # above_threshold_hyp[0] = battery_charge_hyp[0] - threshold if battery_charge[0] > threshold else 0

    for i in range(0, len(tf)-1):
        energy_added[i] = interval * solar_input[i] / 1000  # Convert charging power from Watts to kWh
        energy_consumed[i] = interval * (other_output[i] + mining_output[i]) / 1000
        battery_charge[i+1] = battery_charge[i] + energy_added[i] - energy_consumed[i]
        battery_charge_hyp[i+1] = battery_charge_hyp[i] + energy_added[i] - energy_consumed[i]
        
        if battery_charge[i+1] > max_battery_charge_kwh:
            excess_energy[i] = battery_charge[i+1] - max_battery_charge_kwh
            battery_charge[i+1] = max_battery_charge_kwh
        
        if battery_charge[i+1] > threshold:
            above_threshold[i+1] = battery_charge[i+1] - threshold if battery_charge[i+1] > threshold else 0
            assert above_threshold[i+1] >= 0, f"battery_charge[{i}+1] = {battery_charge[i+1]}, threshold {threshold}"
            # Bad definition
            # above_threshold_hyp[i] = battery_charge_hyp[i+1] - np.sum(excess_energy[0:i-1]) - threshold if battery_charge_hyp[i+1] - threshold > 0 else 0
        
        if battery_charge[i+1] < 0:
            raise ValueError(f"Battery energy dropped below 0!\n{battery_charge}")
        if battery_charge_hyp[i+1] < 0:
            raise ValueError(f"Battery energy hyp dropped below 0!\n{battery_charge_hyp}")
    
    # return battery_charge, excess_energy, above_threshold
    return battery_charge, excess_energy, above_threshold, battery_charge_hyp


def calculate_battery_charge_new(interval: float, tf: np.ndarray, battery_params: dict, solar_input: np.ndarray, other_output: np.ndarray, mining_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the battery charge levels over the prediction timeframe

    ### Arguments:
        - interval {float}: [h] duration of each timeframe, in hours
        - tf {np.ndarray}: numpy array of the timeframes
        - battery_params {dict}: dict of battery parameters
        - solar_input {np.ndarray}: [W] numpy array of the solar power generated
        - other_output {np.ndarray}: [W] numpy array of the power used by non-mining devices
        - mining_output {np.ndarray}: [W] numpy array of the power used by mining equipment
    
    ### Raises:
        - ValueError: if the battery charge drops below 0 Wh at any point
    
    Returns:
        - battery_charge {np.ndarray}: [kWh] numpy array of the battery charge levels
        - excess_energy {np.ndarray}: [kWh] numpy array of the generated energy that exceeds the battery capacity
        - above_threshold {np.ndarray}: [kWh] numpy array of the battery energy above the usable threshold
    """
    initial_battery_charge = battery_params['initial_battery_charge']
    threshold = battery_params['threshold']
    max_battery_charge_kwh = battery_params['max_battery_charge_kwh']
    battery_charge = np.zeros_like(tf, dtype=float)
    battery_charge[0] = initial_battery_charge  # Initial value of battery charge in kWh
    battery_charge_hyp = np.zeros_like(tf, dtype=float)
    battery_charge_hyp[0] = initial_battery_charge  # Initial value of battery charge in kWh
    excess_energy = np.zeros_like(tf, dtype=float)
    above_threshold = np.zeros_like(tf, dtype=float)
    above_threshold[0] = battery_charge[0] - threshold if battery_charge[0] > threshold else 0
    # above_threshold_hyp = np.zeros_like(tf, dtype=float)
    # above_threshold_hyp[0] = battery_charge_hyp[0] - threshold if battery_charge[0] > threshold else 0

    energy_change = (solar_input - other_output - mining_output) / 1000 * interval

    for i in range(0, len(tf)-1):
        battery_charge[i+1] = battery_charge[i] + energy_change[i]
        battery_charge_hyp[i+1] = battery_charge_hyp[i] + energy_change[i]
        
        if battery_charge[i+1] > max_battery_charge_kwh:
            excess_energy[i] = battery_charge[i+1] - max_battery_charge_kwh
            battery_charge[i+1] = max_battery_charge_kwh
        
        if battery_charge[i+1] > threshold:
            above_threshold[i+1] = battery_charge[i+1] - threshold if battery_charge[i+1] > threshold else 0
            # Bad definition
            # above_threshold_hyp[i] = battery_charge_hyp[i+1] - np.sum(excess_energy[0:i-1]) - threshold if battery_charge_hyp[i+1] - threshold > 0 else 0
        
        if battery_charge[i+1] < 0:
            raise ValueError(f"Battery energy dropped below 0!\n{battery_charge}")
        if battery_charge_hyp[i+1] < 0:
            raise ValueError(f"Battery energy hyp dropped below 0!\n{battery_charge_hyp}")
    
    # return battery_charge, excess_energy, above_threshold
    return battery_charge, excess_energy, above_threshold, battery_charge_hyp


def charge_subrange_split(battery_charge: np.ndarray, battery_params: dict):
    # Find the indices of elements above the threshold
    # threshold = battery_params['minimum_battery_charge_NE']  # Let the battery drop down to the limit_NE (except for the last interval?)
    threshold = battery_params['threshold']
    indices = np.nonzero(battery_charge >= threshold)[0]
    subranges = []
    sublist = []
    if len(indices) == 0:
        # print("indices is empty!")
        subranges = []
        return subranges
    else:
        sublist.append(indices[0])
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                subranges.append(sublist)
                sublist = []
            sublist.append(indices[i])
        if sublist:
            subranges.append(sublist) # For the last one
    return subranges


# def calculate_bucket_availability(buckets: list[EnergyBucket], system_params: dict) -> list[EnergyBucket]:
def calculate_bucket_availability(buckets: list[EnergyBucket], battery_charge: np.ndarray, available_power: np.ndarray, excess_energy: np.ndarray, system_params: dict) -> list[EnergyBucket]:
    for i, bucket in enumerate(buckets):
        bucket.available = min(interval * available_power[i] / 1000,
                            interval * maximum_rigs_power / 1000,
                            # energy_rectangles[i] + threshold - minimum_battery_charge_NE,
                            #    energy_added[i],
                            #    above_threshold_hyp[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0)
                            # (battery_charge[i] - threshold) + excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE else 0
                               above_threshold[i] + excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0
                            #    excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0)
                            #    battery_charge_hyp[i] - threshold if battery_charge_hyp[i] > minimum_battery_charge_NE > 0 else 0)
                            #    battery_charge[i] - threshold if battery_charge[i] > minimum_battery_charge_NE else 0)
                            #    battery_charge[i] + excess_energy[i] - threshold if above_threshold[i] > 0 else 0)
        )

        if bucket.available < 0:
            # print("Bucket {} is negative! {}, {}, {}".format(i,
            raise ValueError("Bucket {} is negative! {}, {}, {}".format(i,
                                                            interval * available_power[i] / 1000,
                                                            interval * maximum_rigs_power / 1000,
                                                            # (threshold - minimum_battery_charge_NE) + excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0))
                                                            # (battery_charge[i] - threshold) + excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0))
                                                            above_threshold[i] + excess_energy[i] if battery_charge[i] > minimum_battery_charge_NE > 0 else 0))

        if bucket.available == interval * available_power[i] / 1000:
            bucket.power_limited = True
            bucket.limitation = Limitation.POWER
        elif bucket.available == interval * maximum_rigs_power / 1000:
            bucket.miner_limited = True
            bucket.limitation = Limitation.MINER
        else:
            bucket.limitation = Limitation.ENERGY
    
    return buckets


def energy_redistribution(
    system_params: dict,
    battery_charge: np.ndarray,
    excess_energy: np.ndarray,
    above_threshold: np.ndarray,
    buckets: list[EnergyBucket],
    energies_to_spread: np.ndarray
    ):

    interval = system_params['interval']
    mining_power = np.zeros_like(tf, dtype=float)

    # Sort the 'buckets' array by the 'value' property in descending order using a stable sorting algorithm
    sorted_buckets = sorted(buckets, key=lambda bucket: bucket.value, reverse=True)
    sorted_buckets_indices = [buck.index for buck in sorted_buckets]
    
    # Undistributed energy: means it exceeded the buckets' availability
    undistributed_energy = np.zeros_like(energies_to_spread, dtype=float)
    num_energies = len(energies_to_spread)
    for i in range(0, num_energies):
        excess = np.copy(energies_to_spread[i])
        # if i == num_energies: energies_to_spread[i] = battery_charge[-1] - threshold  # Recalculate at the end?
        print(f"Excess energy {i}: {excess}")
        if excess > 0:
            # for bucket in sorted_buckets:
            for idx in sorted_buckets_indices:
                # print(f"bucket.index: {bucket.index}, i: {i}")
                # idx = bucket.index
                if idx <= i:
                # if idx < i or i == 0:
                    diff = buckets[idx].available - buckets[idx].filled
                    # diff = buckets[idx].available
                    if diff > 0:
                        # print(f"Bucket {idx} has available {buckets[idx].available} and diff {diff}")
                        if diff > excess:
                            # try:
                            buckets[idx].filled += excess
                            # for b in buckets[idx:i]: b.available -= excess
                            mining_power[idx] += excess / interval * 1000
                            # print(f"Adding {excess} kWh to bucket[{idx}].filled: {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                            print(f"Adding {excess} kWh to bucket[{idx}] with diff: {diff} - bucket[{idx}].available: {buckets[idx].available} - battery_charge[{idx}]: {battery_charge[idx]}")
                            # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            # if np.min(battery_charge) < threshold:
                            if np.min(battery_charge) < minimum_battery_charge_NE:
                                # print(f"Battery charge dropped below minimum threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                print(f"Battery charge dropped to {battery_charge[idx]} below minimum threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - battery_charge[{idx}]: {battery_charge[idx]} - mining_power[{idx}]: {mining_power[idx]}")
                                buckets[idx].filled -= excess
                                # for b in buckets[idx:i]: b.available += excess
                                mining_power[idx] -= excess / interval * 1000
                                # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                print(f"After: bucket[{idx}].filled {buckets[idx].filled} - battery_charge[{idx}]: {battery_charge[idx]} - mining_power[{idx}]: {mining_power[idx]}")
                                continue  # Restore the current bucket and try the next one
                            else:
                                available_power = battery_output + solar_input - other_output - mining_power
                                print(f"Battery final: {battery_charge[-1]}")
                                if np.min(available_power) < 0 or np.max(available_power) > battery_output + np.max(solar_input):
                                    print("available_power:\n", available_power)
                                    print("solar_input:\n", solar_input)
                                    print("other_output:\n", other_output)
                                    print("mining_power:\n", mining_power)
                                    raise ValueError
                                # buckets = calculate_bucket_availability(buckets, battery_charge, available_power, excess_energy, system_params)
                            excess = 0  # To keep an accurate track of the undistributed energy
                            break  # If successful, then we're done with the current energy excess
                            # print(f"Filling bucket {bucket.index} of value {bucket.value:.5f} and diff {diff:.5f} with index {i} of content {excess:.5f}")
                        else:
                            print(f"Adding {diff} kWh (of {excess}) to bucket[{idx}] - bucket[{idx}].available: {buckets[idx].available} - battery_charge[{idx}]: {battery_charge[idx]}")
                            buckets[idx].filled += diff
                            # for b in buckets[idx:i]: b.available -= diff
                            excess -= diff
                            mining_power[idx] += diff / interval * 1000
                            # print(f"Adding {diff} kWh to bucket[{idx}].filled: {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                            # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            # if np.min(battery_charge) < threshold:
                            if np.min(battery_charge) < minimum_battery_charge_NE:
                                # print(f"Battery charge dropped below minimum threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                print(f"Battery charge dropped to {battery_charge[idx]} below minimum threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - battery_charge[{idx}]: {battery_charge[idx]}")
                                buckets[idx].filled -= diff
                                # for b in buckets[idx:i]: b.available += diff
                                excess += diff
                                mining_power[idx] -= diff / interval * 1000
                                # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                print(f"After: bucket[{idx}].filled {buckets[idx].filled} - battery_charge[{idx}]: {battery_charge[idx]}")
                                continue
                            else:
                                available_power = battery_output + solar_input - other_output - mining_power
                                print(f"Battery final: {battery_charge[-1]}")
                                if np.min(available_power) < 0 or np.max(available_power) > battery_output + np.max(solar_input):
                                    print("available_power:\n", available_power)
                                    print("solar_input:\n", solar_input)
                                    print("other_output:\n", other_output)
                                    print("mining_power:\n", mining_power)
                                    raise ValueError
                                # buckets = calculate_bucket_availability(buckets, battery_charge, available_power, excess_energy, system_params)
                            # # print(f"Filling bucket {bucket.index} of value {bucket.value:.5f} and diff {diff:.5f} with index {i} of content {diff:.5f}, leaving {excess:.5f}")
                    else:
                        raise ValueError(f"diff[{idx}] < 0!! {diff}")
            # if excess < 0:
            #     print(f"Excess {i} = {excess} is < 0 !!!")
            assert excess >= 0, f"Excess {i} = {excess} is < 0 !!!"
            # elif excess > 0:
            if excess > 0:
                undistributed_energy[i] += excess
                print(f"Excess {i} = {excess} has not been distributed!!! - undistributed_energy[{i}] = {undistributed_energy[i]}")
            else:
                print(f"Excess {i} = {excess}")

    return buckets, battery_charge, excess_energy, above_threshold, undistributed_energy


if __name__ == "__main__":
    # Create an array of integers from 0 to 71 representing hours
    interval = 1  # Time interval in hours
    tf = np.arange(0, 72, interval)
    initial_battery_charge = 7  # [kWh] Initial battery charge
    battery_output = 2500  # [W] Maximum power deliverable by the battery
    maximum_rigs_power = 2500  # [W] Total power of the mining rigs
    max_battery_charge_kwh = 10  # [kWh] Maximum battery charge
    threshold = 6  # [kWh] Battery threshold value
    minimum_battery_charge_NE = 4  # [kWh] Minimum battery charge, for safety
    system_params = {
        'interval': interval,
        'initial_battery_charge': initial_battery_charge,
        'threshold': threshold,
        'max_battery_charge_kwh': max_battery_charge_kwh,
        'minimum_battery_charge_NE': minimum_battery_charge_NE,
        'battery_output': battery_output,
        'maximum_rigs_power': maximum_rigs_power,
    }

    # def calculate_battery_charge wuz here

    # class EnergyBucket wuz here


    # Create an array of floats following a sinusoidal pattern for solar input
    solar_amplitude = 1200  # Amplitude of oscillation for solar input
    solar_center = 1200  # Center value for solar input
    solar_input = solar_center + solar_amplitude * np.sin(2 * np.pi * tf / 24)
    # solar_input = np.ones_like(tf, dtype=float) * solar_amplitude
    solar_energy_input = interval * solar_input / 1000

    # Create an array of floats following a cosine pattern for other output
    other_amplitude = 450  # Amplitude of oscillation for other output
    other_center = 450  # Center value for other output
    other_output = other_center + other_amplitude * np.cos(5 * 2 * np.pi * tf / 24)
    # other_output = np.ones_like(tf, dtype=float) * other_amplitude
    other_energy_output = interval * other_output / 1000

    # Calculate the charging power by subtracting other output from solar input
    charging_power = solar_input - other_output
    charging_energy = charging_power / 1000 * interval

    # Calculate the available power
    available_power = battery_output + charging_power

    # Check if available_power ever drops below 0
    if np.min(available_power) < 0:
        raise ValueError("Something went horribly wrong! available_power < 0!")

    # Calculate the battery charge by accumulating the energy added by the charging power
    battery_charge = np.zeros_like(tf, dtype=float)
    battery_charge[0] = initial_battery_charge  # Initial value of battery charge in kWh
    # battery_charge_hyp = np.zeros_like(tf, dtype=float)
    # battery_charge_hyp[0] = initial_battery_charge  # Initial value of battery charge in kWh

    print(f"Theoretical energy availability:\n",
        f"Initial charge {battery_charge[0]} kWh - threshold {threshold} kWh\n",
        f"Sum solar_energy_input: {np.sum(solar_energy_input)} kWh\n",
        f"Sum other_energy_output: {np.sum(other_energy_output)} kWh\n",
        f"Net sum: {battery_charge[0] - threshold + np.sum(solar_energy_input) - np.sum(other_energy_output)}\n")

    # Create an array to store the excess energy
    excess_energy = np.zeros_like(tf, dtype=float)

    # Create an array to store the energy added
    energy_added = np.zeros_like(tf, dtype=float)

    # Create a variable to store the difference between battery charge and threshold
    # above_threshold = np.zeros_like(tf, dtype=float)
    # above_threshold_hyp = np.zeros_like(tf, dtype=float)  # All energy above the threshold, including excess

    # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, np.zeros_like(tf, dtype=float))
    battery_charge, excess_energy, above_threshold, battery_charge_hyp = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, np.zeros_like(tf, dtype=float))
    for val in above_threshold:
        assert val >= 0
    original_battery_charge = np.copy(battery_charge)
    original_excess_energy= np.copy(excess_energy)

    print("original_battery_charge:\n", original_battery_charge)

    energy_rectangles = np.zeros_like(tf, dtype=float)
    energy_rectangles[-1] = battery_charge[-1] - threshold
    for i in range(len(tf) - 2, -1, -1):
        energy_rectangles[i] = min(energy_rectangles[i+1], battery_charge[i] - threshold)


    # Create an array of floats following a sinusoidal pattern for the 'value' property of the buckets
    value_amplitude = 0.0002  # Amplitude of oscillation for 'value'
    value_center = 0.01  # Center value for 'value'
    value_frequency = 10  # Frequency of oscillation for 'value'
    value_ramp = np.copy(tf) * (-0.00002)
    value = np.zeros_like(tf, dtype=float)
    # value = value_center + value_amplitude * np.cos(2 * np.pi * np.arange(72) / value_frequency)
    # value = value_center + value_amplitude * np.cos(2 * np.pi * np.arange(72) / value_frequency)
    value -= value_ramp

    # Print the 'value' array
    print("value:\n", value)

    # Create an array of EnergyBucket objects
    buckets = [EnergyBucket(i, value) for i, value in enumerate(value)]

    # Calculate the 'available' value of each bucket
    # def calculate_bucket_availability wuz here

    # buckets = calculate_bucket_availability(buckets, system_params)
    buckets = calculate_bucket_availability(buckets, battery_charge, available_power, excess_energy, system_params)
    for bucket in buckets:
        assert bucket.available >= 0, f"Bucket[{bucket.index}].available = {bucket.available}"

    # Calculate the cumulative sum of 'bucket.available' values before sorting
    available_energy = np.array([bucket.available for bucket in buckets])
    original_available_energy = np.copy(available_energy)
    cumulative_sum_available = np.cumsum(available_energy)

    # Sort the 'buckets' array by the 'value' property in descending order using a stable sorting algorithm
    sorted_buckets = sorted(buckets, key=lambda bucket: bucket.value, reverse=True)

    sorted_buckets_indices = [buck.index for buck in sorted_buckets]
    print(f"Sorted bucket indices: {sorted_buckets_indices}")

    # Set the 'filled' values of each bucket according to excess energy
    mining_power = np.zeros_like(tf, dtype=float)
    energies_to_spread = above_threshold
    # energies_to_spread = np.append(excess_energy, above_threshold[-1])
    # energies_to_spread = np.append(available_energy, above_threshold[-1])
    print("Energies to spread:\n", energies_to_spread)
    print("Cumulative energies to spread:\n", np.cumsum(energies_to_spread))
    print("Sum of energies to spread:", np.sum(energies_to_spread))

    # Undistributed energy: means it exceeded the buckets' availability
    undistributed_energy = np.zeros_like(energies_to_spread, dtype=float)

    for i in range(0, len(energies_to_spread)):
        assert energies_to_spread[i] >= 0, f"energies_to_spread[{i}] == {energies_to_spread[i]} < 0"
        if i == 0:
            excess = energies_to_spread[i]
            print(f"Excess energy {i}: {excess}")
        elif i > 0:
            excess = energies_to_spread[i] + undistributed_energy[i-1]
            print(f"Excess energy {i}: {energies_to_spread[i]} + {undistributed_energy[i-1]} undistributed")
        if excess > 0:
            # for bucket in sorted_buckets:
            for idx in sorted_buckets_indices:
                # print(f"bucket.index: {bucket.index}, i: {i}")
                # idx = bucket.index
                # if idx <= i:  # Try to spend energy ahead of time
                    diff = buckets[idx].available - buckets[idx].filled
                    if diff > 0:
                        print(f"Bucket {idx} has available {buckets[idx].available} and diff {diff}")
                        if diff > excess:
                            # try:calculate_bucket_availability
                            buckets[idx].filled += excess
                            mining_power[idx] += excess / interval * 1000
                            # print(f"Adding {excess} kWh to bucket[{idx}].filled: {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                            print(f"Adding {excess} kWh to bucket[{idx}] with diff: {diff} - bucket[{idx}].available: {buckets[idx].available} - battery_charge[{idx}]: {battery_charge[idx]}")
                            # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, mining_power)
                            battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            # buckets = calculate_bucket_availability(buckets)
                            # except ValueError:
                            if np.min(battery_charge) < minimum_battery_charge_NE:
                                print(f"Battery charge dropped below safety threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                buckets[idx].filled -= excess
                                mining_power[idx] -= excess / interval * 1000
                                # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, mining_power)
                                battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                # buckets = calculate_bucket_availability(buckets)
                                print(f"After: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                continue  # Restore the current bucket and try the next one
                            else:
                                # energies_to_spread[i:] -= excess
                                available_power = battery_output + solar_input - other_output - mining_power
                                print(f"Battery final: {battery_charge[-1]}")
                                if np.min(available_power) < 0 or np.max(available_power) > battery_output + np.max(solar_input):
                                    print("available_power:\n", available_power)
                                    print("solar_input:\n", solar_input)
                                    print("other_output:\n", other_output)
                                    print("mining_power:\n", mining_power)
                                    raise ValueError
                            # if battery_charge[-1] < threshold:
                            #     print(f"Final battery charge drops below threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                            excess = 0  # To keep an accurate track of the undistributed energy
                            break  # If successful, then we're done with the current energy excess
                        else:
                            buckets[idx].filled += diff
                            excess -= diff
                            mining_power[idx] += diff / interval * 1000
                            # print(f"Adding {diff} kWh to bucket[{idx}].filled: {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                            print(f"Adding {diff} kWh (of {excess}) to bucket[{idx}] - bucket[{idx}].available: {buckets[idx].available} - battery_charge[{idx}]: {battery_charge[idx]}")
                            # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, mining_power)
                            battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                            # buckets = calculate_bucket_availability(buckets)
                            # except ValueError:
                            if np.min(battery_charge) < minimum_battery_charge_NE:
                                print(f"Battery charge dropped below minimum threshold, reverting.\nBefore: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                buckets[idx].filled -= diff
                                excess += diff
                                mining_power[idx] -= diff / interval * 1000
                                # battery_charge, excess_energy, above_threshold = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                battery_charge, excess_energy, above_threshold, _ = calculate_battery_charge(interval, tf, system_params, solar_input, other_output, mining_power)
                                # buckets = calculate_bucket_availability(buckets)
                                print(f"After: bucket[{idx}].filled {buckets[idx].filled} - mining_power[{idx}]: {mining_power[idx]}")
                                # continue  # Not needed because we're continuing to the next bucket anyway
                            else:
                                # energies_to_spread[i:] -= diff
                                available_power = battery_output + solar_input - other_output - mining_power
                                print(f"Battery final: {battery_charge[-1]}")
                                if np.min(available_power) < 0 or np.max(available_power) > battery_output + np.max(solar_input):
                                    print("available_power:\n", available_power)
                                    print("solar_input:\n", solar_input)
                                    print("other_output:\n", other_output)
                                    print("mining_power:\n", mining_power)
                                    raise ValueError
                    elif diff == 0:
                        print(f"diff[{idx}] == 0")
                    else:
                        raise ValueError(f"diff[{idx}] < 0!! {diff}")
                # else:  # Try to keep energy for later
                #     print(f"Trying to keep energy_to_spread[{i}] = {excess} instead of spending it on bucket {idx}")
                #     mining_power[idx] = 0
                    # diff = buckets[idx].available - buckets[idx].filled
                    # if diff > 0:
                    #     print(f"Bucket {idx} has available {buckets[idx].available} and diff {diff}")
                    #     if diff > excess:
                    #     else:
                    # pass
            assert excess >= 0, f"Excess {i} = {excess} is < 0 !!!"
            if excess > 0:
                undistributed_energy[i] += excess
                print(f"Excess {i} = {excess} has not been distributed!!!")
            else:
                print(f"Excess {i} = {excess}")

    print("undistributed_energy:\n", undistributed_energy)
    print("sum of undistributed_energy:", np.sum(undistributed_energy))

    # Re-sort the 'sorted_buckets' array by the 'index' property in ascending order using a stable sorting algorithm
    # resorted_buckets = sorted(sorted_buckets, key=lambda bucket: bucket.index, reverse=True)

    cumulative_sum_filled = np.cumsum([bucket.filled for bucket in buckets])
    # cumulative_sum_filled = np.cumsum([bucket.filled for bucket in resorted_buckets])

    # Print the 'value', 'available', and 'power_limited' of each bucket in the sorted order
    # Print the 'value' of each bucket in the sorted order
    # for bucket in sorted_buckets:
    print("SORTED BUCKETS")
    for bucket in sorted(buckets, key=lambda bucket: bucket.value, reverse=True):
        # print(f"Index: {bucket.index:2d}", f"Value: {bucket.value:.5f}", f"Available: {bucket.available:.5f}", f"Filled: {bucket.filled:.5f}", "Power Limited:", bucket.power_limited, "Miner Limited:", bucket.miner_limited)
        print(f"Index: {bucket.index:2d}", f"Value: {bucket.value:.5f}", f"Available: {bucket.available:.5f}", f"Filled: {bucket.filled:.5f}", f"Limitation: {bucket.limitation.key}")
        assert bucket.filled <= bucket.available
        assert bucket.filled >= 0
    # print("UNSORTED BUCKETS")
    # for bucket in buckets:
    #     print(f"Index: {bucket.index:2d}", f"Value: {bucket.value:.5f}", f"Available: {bucket.available:.5f}", f"Filled: {bucket.filled:.5f}", "Power Limited:", bucket.power_limited, "Miner Limited:", bucket.miner_limited)
    #     print(f"Index: {bucket.index:2d}", f"Value: {bucket.value:.5f}", f"Available: {bucket.available:.5f}", f"Filled: {bucket.filled:.5f}", f"Limitation: {bucket.limitation}")
    #     assert bucket.filled <= bucket.available
    #     assert bucket.filled >= 0

    # Print the cumulative sum of 'bucket.available' values before sorting
    print("Cumulative Sum Available:\n", cumulative_sum_available)
    print("Cumulative Sum Filled:\n", cumulative_sum_filled)

    print("Mining Power:\n", mining_power)
    print("Mining Power Sum:", np.sum(mining_power))

    print("initial_battery_charge + np.sum(charging_energy) - cumulative_sum_filled[-1]: ", initial_battery_charge + np.sum(charging_energy) - cumulative_sum_filled[-1])
    print("minimum_battery_charge_NE: ", minimum_battery_charge_NE)
    # Initial energy + solar input - other output - mining output must be >= than the minimum battery charge in the end
    assert initial_battery_charge + np.sum(charging_energy) - cumulative_sum_filled[-1] >= minimum_battery_charge_NE
    assert_almost_equal(initial_battery_charge + np.sum(charging_energy[:-1]) - cumulative_sum_filled[-1] - np.sum(excess_energy[:-1]), battery_charge[-1])


    # Plot powers
    fig, ax = plt.subplots()
    plt.plot(tf, solar_input, color='blue', label='solar input [W]')
    plt.plot(tf, other_output, color='red', label='other output [W]')
    plt.plot(tf, charging_power, color='green', label='charging power [W]')
    plt.plot(tf, available_power, color='black', label='available power [W]')
    plt.plot(tf, mining_power, color='brown', label='mining power [W]')
    ax.set_xlabel("Hours [h]")
    ax.set_ylabel("Power [W]")
    ax.set_title("Power [W]")
    plt.ylim([min(0, min(charging_power)), max([max(solar_input), max(other_output), max(available_power), max(mining_power)])])
    # ax.set_yscale('log')
    ax.grid(which='both')
    ax.legend()
    plt.show(block=False)

    # Plot energies
    energy_available = np.array([bucket.available for bucket in buckets])
    # for i in range(len(energy_available)):  # TODO: review this?
    #     if energy_available[i] + minimum_battery_charge_NE > battery_charge[i] and battery_charge[i] >= minimum_battery_charge_NE:
    #         print(f"energy_available[{i}] > battery_charge[{i}]")
    energy_filled = np.array([bucket.filled for bucket in buckets])
    energy_diff = np.array([x - y for x, y in zip(energy_available, energy_filled)])
    print("energy_diff:\n", energy_diff)
    if np.min(energy_diff) < 0:
        print("energy_diff drops below 0!")
    battery_final = battery_charge
    print("battery_final:\n", battery_final)

    fig, ax = plt.subplots()
    plt.bar(tf, energy_rectangles, bottom=threshold, label='energy rectangles [kWh]')
    plt.plot(tf, battery_charge, color='blue', label='battery charge [kWh]')
    plt.plot(tf, original_battery_charge, color='blue', label='originalbattery charge [kWh]')
    # plt.bar(tf, -cumulative_sum_filled, bottom=original_battery_charge+excess_energy, label='charge hyp - cumulative sum filled [kWh]')
    # plt.plot(tf, battery_charge_hyp, color='blue', label='battery charge [kWh]')
    plt.plot(tf, energy_added, color='green', label='energy added [kWh]')
    plt.plot(tf, excess_energy, color='red', label='excess energy [kWh]')
    plt.scatter(tf, excess_energy, color='red', label='excess energy [kWh]')
    plt.plot(tf, original_excess_energy, color='red', label='original excess energy [kWh]')
    plt.plot(tf, energy_available, color='black', label='energy available [kWh]')
    plt.plot(tf, energy_filled, color='purple', label='energy filled [kWh]')
    plt.plot(tf, [e + threshold for e in energy_filled], color='purple', label='energy filled [kWh]')
    # plt.plot(tf, cumulative_sum_filled, color='purple', label='cumulative energy filled [kWh]')
    plt.plot(tf, cumulative_sum_filled, label='cumulative energy filled [kWh]')
    plt.scatter(tf, battery_final, color='brown', label='battery final [kWh]')
    # plt.plot(tf, above_threshold_hyp, color='red', label='above threshold hyp [kWh]')
    plt.axhline(minimum_battery_charge_NE, color='red', label='minimum battery charge [kWh]')
    ax.set_xlabel("Hours [h]")
    ax.set_ylabel("Charge [kWh]")
    ax.set_title("Energy [kWh]")
    # plt.ylim([min(0, min(battery_charge), min(battery_final)), max(max(battery_charge_hyp), max(battery_final))])
    plt.ylim([min(0, min(battery_charge), min(battery_final)), max(max(battery_charge), max(battery_final), max_battery_charge_kwh)])
    # ax.set_yscale('log')
    ax.grid(which='both')
    ax.legend()
    plt.show(block=False)

    # Check values
    values = [bucket.value for bucket in buckets]

    fig, ax = plt.subplots()
    plt.scatter(tf, value, color='blue', label='value')
    plt.scatter(tf, values, color='red', label='values')
    ax.set_xlabel("Hours [h]")
    ax.set_ylabel("Values")
    ax.set_title("Values")
    # plt.ylim([0, max(values)])
    # ax.set_yscale('log')
    ax.grid(which='both')
    ax.legend();
    plt.show()
