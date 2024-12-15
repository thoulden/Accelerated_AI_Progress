import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    # === Single Simulation Code ===
    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')
    
    # Option to compute growth
    compute_growth = st.sidebar.checkbox('Compute Growth', value=True)

    # Parameters for the simulation
    lambda_sample = st.sidebar.number_input('Parallelizability (λ)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    r_0_sample = st.sidebar.number_input('Initial Research Productivity (r₀)', min_value=0, max_value=5.0, value=0.7, step=0.1)
    Yr_Left_sample = st.sidebar.number_input('Years Till Ceiling', min_value=1, max_value=50, value=9, step=0.5, format="%.0e")
    f_sample = st.sidebar.number_input('Acceleration term (f)', min_value=1.0, max_value=100.0, value=8.0, step=0.1)

    if run_simulation:
        def choose_parameters():
    """
    Choose initial parameters manually (no random sampling).
    Returns:
        r_initial: The initial value of r (diminishing returns).
        initial_doubling_time: Initial doubling time in months.
        limit_years: The limit expressed as years of progress at recent rates.
        lambda_factor: The lambda factor for adjusting doubling time.
    """
    # Set the parameters to your desired values here:
    r_initial = r_0_sample
    initial_boost = f_sample
    initial_doubling_time = 3 / initial_boost  # Assume current software doubling time is 3 months
    limit_years = Yr_Left_sample
    lambda_factor = lambda_sample

    return r_initial, initial_doubling_time, limit_years, lambda_factor


    def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, stop_doubling_time=6, lambda_factor=0.5):
    # Convert limit_years into the actual ceiling
        ceiling = 256 ** limit_years

        r = r_initial
        doubling_time = initial_doubling_time
        size = 1.0  # Starting size
        sizes = [size]
        times = [0]  # times in months
        rs = [r]

        # Compute total_doublings
        total_doublings = int(limit_years * 8)
        k = r_initial / total_doublings  # Constant reduction in r per doubling

        time_elapsed = 0  # Track time in months

        while size < ceiling and r > 0 and doubling_time <= stop_doubling_time:
            # Update time
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


    # -------------------------------------------
    # Run a single simulation with chosen values and plot results in log scale
    # -------------------------------------------
    r_initial, initial_doubling_time, limit_years, lambda_factor = choose_parameters()
    times, sizes, rs = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, 6, lambda_factor)

    # Convert times to years
    times_in_years = np.array(times) / 12.0
    sizes = np.array(sizes)

    # Plot software level on a log scale (original plot)
    plt.figure(figsize=(10, 5))
    plt.plot(times_in_years, sizes, label='Software Level')
    plt.xlabel('Time (years)')
    plt.ylabel('Software Level')
    plt.title('Software Level Over Time (Log Scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Now calculate annualized growth rates
    # growth_rate_i = (ln(size_i) - ln(size_{i-1})) / (times_in_years[i] - times_in_years[i-1])
    growth_rates = []
    for i in range(1, len(sizes)):
        dt = times_in_years[i] - times_in_years[i-1]
        if dt > 0:
            rate = (np.log(sizes[i]) - np.log(sizes[i-1])) / dt
            growth_rates.append(rate)
        else:
            # Just in case of zero dt, though it shouldn't happen
            growth_rates.append(np.nan)

    # Corresponding times for growth rates will be midpoints or just t[i], here we use t[i]
    growth_times = times_in_years[1:]

    # Define g and multiples
    g = 2.77
    multipliers = [3, 10, 30]

    # Plot growth rates
    plt.figure(figsize=(10,5))
    plt.plot(growth_times, growth_rates, label='Annualized Growth Rate', color='blue')

    # Add horizontal line for g
    plt.axhline(y=g, color='red', linestyle='--', label=f'g = {g}')

    # Add horizontal lines for multiples of g
    colors = ['green', 'orange', 'purple']
    for m, c in zip(multipliers, colors):
        plt.axhline(y=m*g, color=c, linestyle=':', label=f'{m}x g = {m*g}')

    plt.xlabel('Time (years)')
    plt.ylabel('Annualized Growth Rate')
    plt.title('Annualized Software Growth Rate Over Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
