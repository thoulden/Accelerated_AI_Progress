import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def transform_sizes_to_years(sizes):
    """
    Transform sizes such that 256^n -> n.
    We display AI capabilities in units of "years of progress at recent rates."
    (Assumes that effective AI capabilities are equivalent to a 256× jump per year.)
    """
    return [np.log2(size) / 8 for size in sizes]  # since log2(256) = 8

def plot_single_transformed_simulation(times, sizes, label, Yr_Left_sample):
    """
    Plot a single simulation with transformed sizes.
    
    Parameters:
        times: list of time points in months
        sizes: list of raw sizes
        label: label for the curve
        Yr_Left_sample: a reference ceiling (in years) to plot
    """
    transformed_sizes = transform_sizes_to_years(sizes)
    times_in_years = [t / 12 for t in times]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times_in_years, transformed_sizes, label=label, color='blue', linestyle='-')
    # Plot the "recent pace" line (i.e. y = t in years)
    ax.plot(times_in_years, times_in_years, label='Recent pace of progress', color='black', linestyle=':')
    # Plot the ceiling line for reference
    ax.plot(times_in_years, [Yr_Left_sample] * len(times_in_years), 'black', linewidth=0.5)
    ax.text(times_in_years[2], Yr_Left_sample, 'Ceiling', fontsize=8, color='black')

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('AI capabilities\n(years of progress at 2020-4 pace)', fontsize=12)
    ax.set_title('AI capabilities over time', fontsize=14)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10)
    st.pyplot(fig)
    st.markdown("*Note:* 3 years of progress was roughly the time from GPT-2 to ChatGPT")

def run():
    # Initialize session state for auto-run:
    if "initial_run_done" not in st.session_state:
        st.session_state.initial_run_done = False

    # The "Run Simulation" button.
    run_simulation = st.sidebar.button('Run Simulation')

    # Simulation parameters
    compute_growth = st.sidebar.checkbox('Gradual Boost')
    if compute_growth:
        f_sample_min = st.sidebar.number_input('Initial speed-up ($f_0$)', min_value=1.0, max_value=1000.0,
                                               value=1.0, step=0.1,
                                               help="How many times faster does software progress become immediately?")
        f_sample_max = st.sidebar.number_input('Max speed-up ($f_{max}$)', min_value=f_sample_min,
                                               max_value=1000.0, value=32.0, step=0.1,
                                               help="Max speed-up after 5 years.")
    else:
        f_sample = st.sidebar.number_input('Initial speed-up ($f$)', min_value=0.0, max_value=1000.0,
                                           value=8.0, step=0.1,
                                           help="How many times faster does software progress become?")
        f_sample_max = f_sample
        f_sample_min = f_sample

    r_0_sample = st.sidebar.number_input('$r$', min_value=0.0, max_value=5.0,
                                         value=1.2, step=0.1,
                                         help="Controls diminishing returns to research.")
    Yr_Left_sample = st.sidebar.number_input('Distance to effective limits on software',
                                              min_value=1.0, max_value=50.0, value=9.0, step=0.5,
                                              help="How far is software from effective limits (in years)?")
    lambda_sample = st.sidebar.number_input('Parallelizability (λ)', min_value=0.01, max_value=1.0,
                                            value=0.3, step=0.01,
                                            help="How many times does the pace double if R&D inputs double?")
    retraining_cost = st.sidebar.checkbox('Retraining Cost')

    def run_the_simulation():
        def choose_parameters():
            """
            Set initial parameters and compute the initial time step.
            Returns a tuple:
              (factor_increase, r_initial, initial_factor_increase_time, limit_years,
               lambda_factor, compute_growth_monthly_rate, f_0, f_max, compute_size_start, compute_max)
            """
            if compute_growth:
                factor_increase = 1.1  # For smooth increases in compute growth scenario.
            else:
                factor_increase = 2    # Use doublings when not growing compute.
            r_initial = r_0_sample
            f_0 = f_sample_min
            f_max = f_sample_max
            compute_size_start = 1
            compute_max = 4096
            compute_doubling_time = 5  # months
            compute_growth_monthly_rate = np.log(2) / compute_doubling_time
            limit_years = Yr_Left_sample
            lambda_factor = lambda_sample
            # Compute an initial time step using a doubling time of 3 months:
            doubling_time_starting = 3  # months
            implied_month_growth_rate = np.log(2) / doubling_time_starting
            time_takes_to_factor_increase = np.log(factor_increase) / implied_month_growth_rate
            initial_factor_increase_time = time_takes_to_factor_increase / (1 + f_0)
            return (factor_increase, r_initial, initial_factor_increase_time, limit_years,
                    lambda_factor, compute_growth_monthly_rate, f_0, f_max, compute_size_start, compute_max)

        def dynamic_system(r_initial, initial_factor_increase_time, limit_years, compute_growth_monthly_rate,
                           f_0, f_max, compute_size_start, compute_max, factor_increase, lambda_factor=0.5,
                           max_time_months=72):
            """
            Run the simulation until time_elapsed reaches max_time_months (72 months).
            """
            ceiling = 256 ** limit_years
            r = r_initial
            factor_increase_time = initial_factor_increase_time
            size = 1.0
            compute_size = compute_size_start

            times = [0]
            sizes = [size]
            rs = [r]
            compute_sizes = [compute_size]
            f_values = [f_0]
            f = f_0

            total_factor_increasings = np.log(ceiling) / np.log(factor_increase)
            k = r_initial / total_factor_increasings

            time_elapsed = 0
            while time_elapsed < max_time_months and size < ceiling and r > 0:
                dt = min(factor_increase_time, max_time_months - time_elapsed)
                f_old = f

                time_elapsed += dt
                times.append(time_elapsed)

                # Update size proportionally.
                size *= factor_increase ** (dt / factor_increase_time)
                if size > ceiling:
                    size = ceiling
                sizes.append(size)

                # Update r proportionally.
                r -= k * (dt / factor_increase_time)
                rs.append(r)

                compute_size = compute_size_start * np.exp(compute_growth_monthly_rate * time_elapsed)
                compute_sizes.append(compute_size)

                if compute_size < compute_max:
                    f = f_0 + (f_max - f_0) * (np.log(compute_size / compute_size_start) /
                                               np.log(compute_max / compute_size_start))
                else:
                    f = f_max
                f_values.append(f)

                if r > 0:
                    if retraining_cost:
                        accel_factor = ((lambda_factor * ((1 / r) - 1)) /
                                        (abs(lambda_factor * ((1 / r) - 1)) + 1))
                    else:
                        accel_factor = (lambda_factor * (1 / r - 1))
                    factor_increase_time *= ((factor_increase ** accel_factor) / ((1 + f) / (1 + f_old)))
            return times, sizes, rs, ceiling, compute_sizes, f_values

        (factor_increase, r_initial, initial_factor_increase_time, limit_years,
         lambda_factor, compute_growth_monthly_rate, f_0, f_max, compute_size_start, compute_max) = choose_parameters()

        return dynamic_system(r_initial, initial_factor_increase_time, limit_years, compute_growth_monthly_rate,
                              f_0, f_max, compute_size_start, compute_max, factor_increase, lambda_factor,
                              max_time_months=72)

    # --- Auto-run Logic ---
    # If the user presses the button OR if the simulation has never been run before, run it.
    if run_simulation:
        times, sizes, rs, ceiling, compute_sizes, f_values = run_the_simulation()
        st.session_state.initial_run_done = True
    elif not st.session_state.initial_run_done:
        times, sizes, rs, ceiling, compute_sizes, f_values = run_the_simulation()
        st.session_state.initial_run_done = True
    else:
        st.write("Press **Run Simulation** (in the sidebar) to generate new results.")

    # Display the simulation outputs (if they exist).
    if "initial_run_done" in st.session_state and st.session_state.initial_run_done:
        plot_single_transformed_simulation(times, sizes, label="AI Capabilities Simulation", 
                                           Yr_Left_sample=Yr_Left_sample)

        times_in_years = [t / 12 for t in times]
        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(times_in_years, rs, label='r(t)', color='magenta')
        ax_r.set_xlabel('Time (years)')
        ax_r.set_ylabel('r')
        ax_r.set_title('r Over Time')
        ax_r.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_r.legend()
        st.pyplot(fig_r)

        if compute_growth:
            fig_f, ax_f = plt.subplots(figsize=(10, 5))
            ax_f.plot(times_in_years, f_values, label='f(t)', color='green')
            ax_f.set_xlabel('Time (years)')
            ax_f.set_ylabel('f')
            ax_f.set_title('Acceleration Factor Over Time')
            ax_f.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_f.legend()
            st.pyplot(fig_f)

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
            ax_growth.axhline(y=m * g, color=c, linestyle=':', label=f'{m}x g = {m * g}')
        ax_growth.set_xlabel('Time (years)')
        ax_growth.set_ylabel('Annualized Growth Rate')
        ax_growth.set_title('Annualized Software Growth Rate Over Time')
        ax_growth.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_growth.legend()
        st.pyplot(fig_growth)
        st.markdown("*Note:* an annual growth rate of 2.77 corresponds to doubling every 3 months.")

if __name__ == "__main__":
    run()
