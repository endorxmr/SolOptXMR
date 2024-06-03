import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from energy_redist import EnergyBucket, calculate_battery_charge, calculate_battery_charge_new, calculate_bucket_availability, energy_redistribution, charge_subrange_split


def cos_wave(center, amplitude, freq, tf):
    wave = center + amplitude * np.cos(freq * 2 * np.pi * tf / 24)
    return wave

def sin_wave(center, amplitude, freq, tf):
    wave = center + amplitude * np.sin(freq * 2 * np.pi * tf / 24)
    return wave

def linear(start, stop, tf: np.ndarray):
    arr = np.linspace(start, stop, num=len(tf))
    return arr

class TestBatteryCharge:
    solar_inputs = [
        (linear, 0, 0),
        (cos_wave, 500, 500, 1),
    ]

    other_outputs = [
        (linear, 0, 0),
        (cos_wave, 50, 50, 5),
        (cos_wave, 250, 250, 5),
    ]
    mining_outputs = [
        (linear, 0, 0),
        (cos_wave, 50, 50, 10),
        (cos_wave, 250, 250, 10),
    ]

    @pytest.fixture(scope="class", params=[0.5, 1])
    def interval(self, request):
        yield request.param

    @pytest.fixture(scope="class", params=[5, 72])
    def tf_end(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def tf(self, request, tf_end, interval):
        yield np.arange(0, tf_end, interval)

    @pytest.fixture(scope="class", params=[2,4,6,8])
    def threshold(self, request):
        yield request.param

    @pytest.fixture(scope="class", params=[1, 7, 10])
    def initial_battery_charge(self, request):
        yield request.param

    @pytest.fixture(scope="class", params=[1, 5, 10, 20])
    def max_battery_charge_kwh(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def battery_params(self, initial_battery_charge, threshold, max_battery_charge_kwh):
        battery_params = {'initial_battery_charge': initial_battery_charge, 'threshold': threshold, 'max_battery_charge_kwh': max_battery_charge_kwh}
        yield battery_params

    @pytest.fixture(scope="class", params=solar_inputs)
    def solar_input(self, request, tf):
        fun, *pars = request.param
        yield fun(*pars, tf)

    @pytest.fixture(scope="class", params=other_outputs)
    def other_output(self, request, tf):
        fun, *pars = request.param
        yield fun(*pars, tf)

    @pytest.fixture(scope="class", params=mining_outputs)
    def mining_output(self, request, tf):
        fun, *pars = request.param
        yield fun(*pars, tf)

    @pytest.fixture(scope="class")
    def bat_chg_calc(self, interval, tf, battery_params, solar_input, other_output, mining_output):
        solar_energy_input = np.array(interval * solar_input / 1000)
        other_energy_output = np.array(interval * other_output / 1000)
        mining_energy_output = np.array(interval * mining_output / 1000)
        energies = np.cumsum(solar_energy_input[:-1] - other_energy_output[:-1] - mining_energy_output[:-1])
        if not np.greater(battery_params["initial_battery_charge"] + energies, 0).all():
            with pytest.raises(ValueError):
                bat_chg_calc = calculate_battery_charge_new(interval, tf, battery_params, solar_input, other_output, mining_output)
            pytest.xfail("Battery charge drops below 0")
        else:
            bat_chg_calc = calculate_battery_charge_new(interval, tf, battery_params, solar_input, other_output, mining_output)
            yield bat_chg_calc

    @pytest.fixture(scope="class")
    def subranges(self, bat_chg_calc, battery_params):
        sr = charge_subrange_split(bat_chg_calc[0], battery_params)
        yield sr


    @pytest.mark.dependency(name="compare_old_and_new")
    def test_compare_old_and_new(self, interval, tf, battery_params, solar_input, other_output, mining_output):
        if battery_params["initial_battery_charge"] > battery_params["max_battery_charge_kwh"]:
            pytest.xfail("Initial battery charge cannot be greater than the maximum charge")
        if battery_params["threshold"] > battery_params["max_battery_charge_kwh"]:
            pytest.xfail("Threshold cannot be greater than the maximum charge")

        solar_energy_input = np.array(interval * solar_input / 1000)
        
        other_energy_output = np.array(interval * other_output / 1000)
        
        mining_energy_output = np.array(interval * mining_output / 1000)

        # Note that the last values are skipped because they would affect the battery charge of the timeframe after tf_end
        sum_of_inputs = np.sum(solar_energy_input[:-1])  # [kWh]
        sum_of_other_outputs = np.sum(other_energy_output[:-1])  # [kWh]
        sum_of_mining_outputs = np.sum(mining_energy_output[:-1])  # [kWh]

        energies = np.cumsum(solar_energy_input[:-1] - other_energy_output[:-1] - mining_energy_output[:-1])
        
        # If the initial battery charge is greater than the sum of energies at any point, that means we drop below 0!
        if not np.greater(battery_params["initial_battery_charge"] + energies, 0).all():
            # with pytest.raises(ValueError, match=r"Battery energy dropped below 0!.*"):
            with pytest.raises(ValueError):
                battery_charge, excess_energy, above_threshold, battery_charge_hyp = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, mining_output)
            with pytest.raises(ValueError):
                battery_charge_new, excess_energy_new, above_threshold_new, battery_charge_hyp_new = calculate_battery_charge_new(interval, tf, battery_params, solar_input, other_output, mining_output)
        else:
            battery_charge, excess_energy, above_threshold, battery_charge_hyp = calculate_battery_charge(interval, tf, battery_params, solar_input, other_output, mining_output)
            battery_charge_new, excess_energy_new, above_threshold_new, battery_charge_hyp_new = calculate_battery_charge_new(interval, tf, battery_params, solar_input, other_output, mining_output)
        
            sum_of_excess_energy = np.sum(excess_energy[:-1])  # [kWh]
            sum_of_excess_energy_new = np.sum(excess_energy_new[:-1])  # [kWh]
            
            assert_almost_equal(battery_params["initial_battery_charge"] + sum_of_inputs - sum_of_other_outputs - sum_of_mining_outputs, battery_charge[-1] + sum_of_excess_energy)
            assert_almost_equal(battery_params["initial_battery_charge"] + sum_of_inputs - sum_of_other_outputs - sum_of_mining_outputs, battery_charge_hyp[-1])

            assert_almost_equal(battery_params["initial_battery_charge"] + sum_of_inputs - sum_of_other_outputs - sum_of_mining_outputs, battery_charge_new[-1] + sum_of_excess_energy_new)
            assert_almost_equal(battery_params["initial_battery_charge"] + sum_of_inputs - sum_of_other_outputs - sum_of_mining_outputs, battery_charge_hyp_new[-1])
            assert_almost_equal(battery_charge, battery_charge_new)
            assert_almost_equal(battery_charge_hyp, battery_charge_hyp_new)
            assert_almost_equal(excess_energy, excess_energy_new)
            assert_almost_equal(above_threshold, above_threshold_new)


    @pytest.mark.dependency(depends=["compare_old_and_new"], scope='class')
    def test_charge_subrange_split(self, bat_chg_calc, subranges, battery_params):
        threshold = battery_params['threshold']
        flat_split_indices = [item for sublist in subranges for item in sublist]
        # If an index is in the list of subranges, then we expect that value to be >= the threshold,
        # otherwise we expect it to be below
        for i, bat_chg_val in enumerate(bat_chg_calc[0]):
            if i in flat_split_indices:
                assert bat_chg_val >= threshold
            else:
                assert bat_chg_val < threshold
            assert bat_chg_val > 0  # Strictly > 0 because we xfail bat_chg_calc otherwise
