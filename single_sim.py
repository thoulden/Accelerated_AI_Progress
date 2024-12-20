import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def transform_sizes_to_years(sizes):
    """
    Transform sizes such that 256^n -> n.

    We will display AI capabilities in units of "years of progress at recent rates."
    The sims assume that software has recently doubled every 3 months. We assume
    hardware has recently been contributing an equal amount to AI progress,
    which leaves a doubling time of 1.5 months. That's 8 doublings per year so 256X
    per year.

    More explanation of why these assumptions are reasonable:
    - Compute inputs have doubled every 6 months according to Epoch.
    - Software algorithms have become twice as efficient every ~8 months, but this
    excludes post-training enhancements so we reduce this to 6 months.
    - That's a combined doubling time of 3 months for effective training compute
    (incorporating compute and algorithms).
    - We estimate that each doubling of effective training compute is equivalent
    to ~2 doublings in the parallel size of the AI population, because you
    get smarter models. This model counts as doubling of "AI capabilities" as a doubling
    of the size of the AI population. So we get two doublings of AI capabilities per
    doubling of effective compute. (Search "Better capabilities" in gdoc appendix.)
    - So AI capabilities have recently been doubling every 1.5 months, according
    to this model.
    """
    return [np.log2(size) / 8 for size in sizes]  # log2(256) = 8

def plot_single_transformed_simulation(times, sizes, label, Yr_Left_sample):
    """
    Plot a single simulation with transformed sizes.

    Parameters:
        times: List of time points in months.
        sizes: List of sizes (pre-transformed).
        label: Label for the simulation line.
        ceiling: Ceiling level to plot as a reference line.
    """
    transformed_sizes = transform_sizes_to_years(sizes)
    times_in_years = [t / 12 for t in times]  # Convert months to years

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times_in_years, transformed_sizes, label=label, color='blue', linestyle='-')

    # Add a reference line for the recent pace of progress
    ax.plot(times_in_years, times_in_years, label='Recent pace of progress', color='black', linestyle=':')

    # Add ceiling line
    ax.plot(times_in_years, [Yr_Left_sample] * len(times_in_years), 'black', linewidth=0.5)
    ax.text(times_in_years[2], Yr_Left_sample, 'Ceiling', fontsize=8, color='black')

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('AI capabilities\n(years of progress at 2020-4 pace)', fontsize=12)
    ax.set_title('AI capabilities over time', fontsize=14)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10)
    st.pyplot(fig)
    st.markdown("*Note:* 3 years of progress at the old rate corresponds to 1 GPT-sized jump")

def run():
    # === Single Simulation Code ===
    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')
    
    

    # Parameters for the simulation
    lambda_sample = st.sidebar.number_input('Parallelizability (λ)', min_value=0.01, max_value=1.0, value=0.4, step=0.01)
    r_0_sample = st.sidebar.number_input('Initial Research Productivity (r₀)', min_value=0.0, max_value=5.0, value=1.2, step=0.1)
    Yr_Left_sample = st.sidebar.number_input('Years Till Ceiling', min_value=1.0, max_value=50.0, value=9.0, step=0.5)
    # Option to compute growth
    
    if compute_growth:
        f_sample = st.sidebar.number_input('Acceleration term (f)', min_value=1.0, max_value=1000.0, value=8.0, step=0.1)
    else:
        f_sample_min = st.sidebar.number_input('Initial Acceleration (f)', min_value=1.0, max_value=1000.0, value=8.0, step=0.1)
        f_sample_max = st.sidebar.number_input('Max Acceleration (f)', min_value=f_sample_min, max_value=1000.0, value=8.0, step=0.1)
    # Checkbox for retraining cost
    compute_growth = st.sidebar.checkbox('Compute Growth')
    retraining_cost = st.sidebar.checkbox('Retraining Cost')
    
    
    if run_simulation:
        def choose_parameters():
            factor_increase = 1.1
            r_initial = r_0_sample
            initial_boost = f_sample
            compute_size_start = 1
            compute_max = 4096
            compute_doubling_time = 3
            initial_doubling_time = 3 / initial_boost
            limit_years = Yr_Left_sample
            lambda_factor = lambda_sample
            return r_initial, initial_doubling_time, limit_years, lambda_factor

        def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, stop_doubling_time=6, lambda_factor=0.5):
            ceiling = 256 ** limit_years
            r = r_initial
            doubling_time = initial_doubling_time
            size = 1.0
            sizes = [size]
            times = [0]
            rs = [r]
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings
            time_elapsed = 0
            while size < ceiling and r > 0 and doubling_time <= stop_doubling_time:
                time_step = doubling_time
                time_elapsed += time_step
                times.append(time_elapsed)
                size *= 2
                sizes.append(size)
                r -= k
                rs.append(r)
                if retraining_cost:
                    doubling_factor = (lambda_factor * ((1 / r) - 1))/(abs(lambda_factor * ((1 / r) - 1) + 1))
                else:
                    doubling_factor = (lambda_factor * (1 / r - 1))
                if r > 0:
                    doubling_time *= 2 ** doubling_factor
            return times, sizes, rs, ceiling

        # Run the simulation
        r_initial, initial_doubling_time, limit_years, lambda_factor = choose_parameters()
        times, sizes, rs, ceiling = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, 6, lambda_factor)

        # Plot transformed simulation
        plot_single_transformed_simulation(times, sizes, label="AI Capabilities Simulation", Yr_Left_sample=Yr_Left_sample)

        # Plot r over time
        times_in_years = [t / 12 for t in times]
        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(times_in_years, rs, label='r(t)', color='magenta')
        ax_r.set_xlabel('Time (years)')
        ax_r.set_ylabel('r')
        ax_r.set_title('r Over Time')
        ax_r.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_r.legend()
        st.pyplot(fig_r)

        # Calculate and plot growth rates
        growth_rates = []
        for i in range(1, len(sizes)):
            dt = times_in_years[i] - times_in_years[i-1]
            if dt > 0:
                rate = (np.log(sizes[i]) - np.log(sizes[i-1])) / dt
                growth_rates.append(rate)
            else:
                growth_rates.append(np.nan)
        growth_times = times_in_years[1:]

        g = 2.77
        multipliers = [3, 10, 30]
        fig_growth, ax_growth = plt.subplots(figsize=(10, 5))
        ax_growth.plot(growth_times, growth_rates, label='Annualized Growth Rate', color='blue')
        ax_growth.axhline(y=g, color='red', linestyle='--', label=f'g = {g}')
        colors = ['green', 'orange', 'purple']
        for m, c in zip(multipliers, colors):
            ax_growth.axhline(y=m*g, color=c, linestyle=':', label=f'{m}x g = {m*g}')
        ax_growth.set_xlabel('Time (years)')
        ax_growth.set_ylabel('Annualized Growth Rate')
        ax_growth.set_title('Annualized Software Growth Rate Over Time')
        ax_growth.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_growth.legend()
        st.pyplot(fig_growth)
    else:
        st.write("Press 'Run Simulation' to view results.")
