import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

def run():

    run_sims = st.sidebar.button("Run Simulations")

def sample_parameters():
    """
    Sample initial parameters from uniform and log-uniform distributions.

    Returns:
        r_initial: The initial value of r (diminishing returns).
        initial_doubling_time: Initial doubling time in months.
        limit_years: The limit expressed as years of progress at recent rates.
        lambda_factor: The lambda factor for adjusting doubling time.
    """
    # Sample initial speed-up from log-uniform distribution (range: 2 to 32)
    initial_boost = np.exp(np.random.uniform(np.log(2), np.log(32)))
    initial_doubling_time = 3 / initial_boost  # Assume current software doubling time is 3 months

    # Sample r from a log-uniform distribution (range: 0.4 to 3.6)
    r_initial = np.exp(np.random.uniform(np.log(0.4), np.log(3.6)))

    # Sample limit_years uniformly from 7 to 14
    limit_years = np.random.uniform(7, 14)

    # Sample lambda factor from log-uniform distribution (range: 0.2 to 0.8)
    lambda_factor = np.exp(np.random.uniform(np.log(0.2), np.log(0.8)))

    return r_initial, initial_doubling_time, limit_years, lambda_factor


def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, stop_doubling_time=6, lambda_factor=0.5):
    """
    Simulate the dynamical system with an adjustable lambda factor and stop once the doubling time exceeds the specified limit.

    Parameters:
        r_initial: Initial value of r.
        initial_doubling_time: Initial doubling time in months.
        limit_years: The limit expressed as years of progress at recent rates.
        stop_doubling_time: Stop the simulation if the doubling time exceeds this limit (in months).
        lambda_factor: Factor to adjust the doubling time (sampled from 0.2 to 0.8).

    Returns:
        times: List of time points (in months).
        sizes: List of system sizes over time. The size represents the level of AI capabilities.
        rs: List of r values over time.
    """
    # Convert limit_years into the actual ceiling
    ceiling = 256 ** limit_years

    r = r_initial
    doubling_time = initial_doubling_time
    size = 1.0  # Starting size
    sizes = [size]
    times = [0]
    rs = [r]

    total_doublings = int(np.log2(ceiling))
    k = r_initial / total_doublings  # Constant reduction in r per doubling

    time_elapsed = 0  # Track time in months

    # Run the simulation until size reaches the ceiling or doubling time exceeds the limit
    while size < ceiling and r > 0 and doubling_time <= stop_doubling_time:
        # Update time based on current doubling time
        time_step = doubling_time
        time_elapsed += time_step
        times.append(time_elapsed)

        # Double the size
        size *= 2
        sizes.append(size)

        # Update r
        r -= k
        rs.append(r)

        # Update the doubling time for the next iteration with lambda adjustment
        if r > 0:
            doubling_time *= 2 ** (lambda_factor * (1 / r - 1))

    return times, sizes, rs

def transform_sizes_to_years(sizes):
    """
    Transform sizes such that 256^n -> n.

    We will display AI capabilities in units of "years of progress at recent rates.
    The sims assume that software has recently doubled every 3 months. We assume
    hardware has recently been contributing an equal amount to AI progress,
    which leaves a doubling time of 1.5 months. That's 8 doublings per year so 256X
    per year.

    More explanation of why these assumptions are reasonble:
    - compute inputs have doubled every 6 months according to Epoch
    - software algorithms have become twice as efficient every ~8 months, but this
    excludes post-training enhancements so we reduce this to 6 months
    - That's a combined doubling time of 3 months for effective training compute
    (incprorating compute and algs)
    - we estimate that each doubling of effective training compute is equivalent
    to ~2 doubling in the parallel size of the AI population, because you
    get smarter models. This model counts as doubling of "AI capabilities" as a doubling
    the size of the AI population. So we get two doublings of AI capabilities per
    doubling of effective compute. (Search "Better capabilities" in gdoc appendix)
    - So AI capabilities have recently been doubling every 1.5 months, according
    to this model.
    """
    return [np.log2(size) / 8 for size in sizes]  # log2(256) = 8

def plot_single_transformed_simulation(times, sizes, label):
    """
    Plot a single simulation with transformed sizes.

    Parameters:
        times: List of time points in months.
        sizes: List of sizes (pre-transformed).
        label: Label for the simulation line.
    """
    transformed_sizes = transform_sizes_to_years(sizes)
    times_in_years = [t / 12 for t in times]  # Convert months to years

def calculate_summary_statistics_binary(times, conditions):
    """
    Calculate whether there are periods where the system achieved a rapid growth.
    Specifically, checks for a given speed-up factor over a specified time window.
    Returns a 'yes' or 'no' for each condition instead of counts.

    Calculates the speed-up factor relative to recent 2020-2024 AI progress, when we
    assume AI capabilities doubled 8 times a year (corresponding to a 1.5 month
    doubling time, see above.

    Parameters:
        times: List of time points (in months) corresponding to when each doubling occurred.
        conditions: List of tuples where each tuple is (time_period, speed_up_factor).

    Returns:
        A dictionary with 'yes' or 'no' indicating if the condition was met for each (time_period, speed_up_factor) pair.
    """
    results = {condition: 'no' for condition in conditions}

    for time_period, speed_up_factor in conditions:
        # Calculate the expected number of doublings in the baseline scenario (8 doublings per year)
        baseline_doublings = (time_period / 12) * 8  # 8 doublings per year
        required_doublings = int(baseline_doublings * speed_up_factor)

        # Check for groups of 'required_doublings' within the specified 'time_period'
        for i in range(len(times) - required_doublings):
            time_span = times[i + required_doublings] - times[i]
            if time_span < time_period:
                results[(time_period, speed_up_factor)] = 'yes'
                break  # Stop checking once the condition is met

    return results

def sample_parameters_batch(n_samples):
    """
    Sample n_samples sets of parameters in a vectorized manner.

    Returns:
        A NumPy array of shape (n_samples, 4) containing sampled parameters:
        r_initial, initial_doubling_time, limit_years, lambda_factor.
    """
    # Log-uniform distribution for initial speed-up (2 to 32)
    initial_boost = np.exp(np.random.uniform(np.log(2), np.log(32), size=n_samples))
    initial_doubling_time = 3 / initial_boost  # Initial doubling time

    # Log-uniform distribution for r (0.4 to 3.6)
    r_initial = np.exp(np.random.uniform(np.log(0.4), np.log(3.6), size=n_samples))

    # Uniform distribution for limit_years (7 to 14)
    limit_years = np.random.uniform(7, 14, size=n_samples)

    # Log-uniform distribution for lambda factor (0.2 to 0.8)
    lambda_factor = np.exp(np.random.uniform(np.log(0.2), np.log(0.8), size=n_samples))

    return np.column_stack((r_initial, initial_doubling_time, limit_years, lambda_factor))

    # Run 1000 sims

# Define periods (in months) and speed-up factors
time_periods = [0.4, 1, 4, 12, 36]  # Corresponding to 4 months, 12 months, and 36 months
speed_up_factors = [3, 6, 10, 30]  # Corresponding to 3X, 6X 10X, and 30X speedups

# Create a list of all combinations of time periods and speed-up factors
from itertools import product
conditions = list(product(time_periods, speed_up_factors))

# Number of simulations
n_sims = 1000

# Run 1,000 simulations with independently sampled parameters
times_matrix = []

for _ in range(n_sims):
    # Sample a parameter set
    r, initial_doubling_time, limit_years, lambda_factor = sample_parameters_batch(1)[0]

    # Run the simulation
    times, _, _ = dynamic_system_with_lambda(
        r, initial_doubling_time, limit_years, lambda_factor=lambda_factor
    )

    # Save the times
    times_matrix.append(times)

# Calculate summary statistics for the batch
batch_summary_statistics = {condition: 0 for condition in conditions}

for times in times_matrix:
    # Get binary statistics for this simulation
    simulation_statistics = calculate_summary_statistics_binary(times, conditions)

    # Update batch summary statistics
    for condition in conditions:
        if simulation_statistics[condition] == 'yes':
            batch_summary_statistics[condition] += 1

# Convert counts to fractions
batch_summary_fractions = {condition: count / n_sims for condition, count in batch_summary_statistics.items()}

# Print the results
print("Fraction of simulations meeting each condition:")
for condition, fraction in batch_summary_fractions.items():
    print(f"Condition {condition}: {fraction:.2%}")
