import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run():
    st.header("Multiple Simulations")

    # User inputs for simulation setup
    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=10000, value=1000, step=100)
    simulation_duration = st.sidebar.number_input("Simulation Duration (years)", min_value=1, max_value=100, value=10)
    dt = 1.0 / 12.0  # 1 month time step (for example)
    g = 2.77
    
    # Allow user to specify multiples
    multiples_input = st.sidebar.text_input("Enter multiples of g (comma-separated)", value="3,10,30")
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]

    # Allow user to specify parameter sampling bounds
    st.sidebar.markdown("### Parameter Sampling Bounds")
    st.sidebar.markdown("#### initial boost (log-uniform)")
    ib_low = st.sidebar.number_input("initial_boost low bound", min_value=0.1, value=2.0)
    ib_high = st.sidebar.number_input("initial_boost high bound", min_value=ib_low, value=32.0)

    st.sidebar.markdown("#### r (log-uniform)")
    r_low = st.sidebar.number_input("r low bound", min_value=0.01, value=0.4)
    r_high = st.sidebar.number_input("r high bound", min_value=r_low, value=3.6)

    st.sidebar.markdown("#### limit_years (uniform)")
    ly_low = st.sidebar.number_input("limit_years low bound", min_value=1.0, value=7.0)
    ly_high = st.sidebar.number_input("limit_years high bound", min_value=ly_low, value=14.0)

    st.sidebar.markdown("#### lambda_factor (log-uniform)")
    lf_low = st.sidebar.number_input("lambda_factor low bound", min_value=0.01, value=0.2)
    lf_high = st.sidebar.number_input("lambda_factor high bound", min_value=lf_low, value=0.8)
    
    run_sims = st.sidebar.button("Run Simulations")

    if run_sims:
        # Run multiple simulations
        # Parameter sampling function
        def sample_parameters():
            # initial_boost from log-uniform(ib_low, ib_high)
            initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high)))
            initial_doubling_time = 3 / initial_boost

            # r_initial from log-uniform(r_low, r_high)
            r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high)))

            # limit_years uniform(ly_low, ly_high)
            limit_years = np.random.uniform(ly_low, ly_high)

            # lambda_factor from log-uniform(lf_low, lf_high)
            lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high)))

            return r_initial, initial_doubling_time, limit_years, lambda_factor

        def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration):
            # Convert limit_years into the ceiling
            ceiling = 256 ** limit_years

            r = r_initial
            size = 1.0  # Starting size
            times = [0.0]
            sizes = [size]
            # Compute total_doublings
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings if total_doublings > 0 else 0

            # We'll stop after simulation_duration years
            max_time = simulation_duration

            # Initial doubling_time based on initial_doubling_time
            # We'll track a "doubling_time" equivalent from initial conditions
            doubling_time = initial_doubling_time

            current_time = 0.0
            while size < ceiling and r > 0 and current_time < max_time:
                # Growth for dt months at rate given by doubling_time
                # original code: size *= 2 every doubling_time
                # continuous growth rate g': g' = ln(2)/doubling_time
                g_prime = np.log(2) / doubling_time
                size = size * np.exp(g_prime * dt)

                current_time += dt
                if current_time > max_time:
                    break
                times.append(current_time)
                sizes.append(size)

                # Fraction of a doubling event that occurred in this dt
                fraction_of_doubling = (g_prime * dt) / np.log(2)
                r -= k * fraction_of_doubling
                if r < 0:
                    r = 0

                if r > 0:
                    # Update doubling_time
                    doubling_time *= 2 ** (lambda_factor * (1 / r - 1))

            return np.array(times), np.array(sizes)

        # Run all simulations and store growth rates
        all_growth_rates = []  # will store (time, growth_rates) for each sim
        # We'll assume all sims produce the same time array length by using a common discretization
        # If not guaranteed, we can store them separately and interpolate.

        # We'll run one sim to get a reference time array (assuming all parameters vary slightly)
        # Actually we'll just store each sim times and handle by min length approach:
        sim_times = None
        sim_sizes_list = []

        for _ in range(num_sims):
            r_initial, initial_doubling_time, limit_years, lambda_factor = sample_parameters()
            times_array, sizes_array = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration)
            sim_sizes_list.append(sizes_array)
            if sim_times is None:
                sim_times = times_array
            else:
                # If times differ in length, cut to min length
                min_len = min(len(sim_times), len(times_array))
                sim_times = sim_times[:min_len]
                sim_sizes_list = [s[:min_len] for s in sim_sizes_list]

        # Convert to array for vectorized ops
        sim_sizes_mat = np.array(sim_sizes_list)  # shape (num_sims, T)
        # times shape (T,)

        # Compute growth rates for each simulation
        # growth_rate_i = (ln(size_i) - ln(size_{i-1})) / (t_i - t_{i-1})
        # We'll do this for i=1,... for each simulation
        growth_times = sim_times[1:]
        sim_growth_rates = []
        for i in range(num_sims):
            sizes_arr = sim_sizes_mat[i]
            gr = (np.log(sizes_arr[1:]) - np.log(sizes_arr[:-1])) / (growth_times[1] - growth_times[0])  # uniform dt
            sim_growth_rates.append(gr)
        sim_growth_rates = np.array(sim_growth_rates)  # (num_sims, T-1)

        # Now we have growth_times and sim_growth_rates
        # For each multiple, compute fraction of sims exceeding multiple*g at each time step
        fractions_dict = {}
        for m in multiples:
            threshold = m * g
            # Check how many exceed threshold at each time step
            exceed_bool = sim_growth_rates > threshold
            frac = exceed_bool.sum(axis=0) / num_sims
            fractions_dict[m] = frac

        # Plot the fractions over time
        fig, ax = plt.subplots(figsize=(10,5))
        for m in multiples:
            ax.plot(growth_times, fractions_dict[m], label=f'{m}x g')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Fraction of Sims Exceeding Threshold')
        ax.set_title('Fraction of Simulations with Growth Rates Above Multiples of g Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        st.pyplot(fig)

        # Produce a table
        # Let's pick a few time points to report fractions at (e.g., 1 year, 2 years, 5 years)
        report_times = [1, 2, 5, simulation_duration] 
        # Interpolate fractions at these times or pick nearest index
        def nearest_index(time_array, val):
            return (np.abs(time_array - val)).argmin()

        data_for_table = {}
        data_for_table['Time (years)'] = report_times
        for m in multiples:
            fractions_at_times = []
            for rt in report_times:
                idx = nearest_index(growth_times, rt)
                fractions_at_times.append(fractions_dict[m][idx])
            data_for_table[f'{m}x g'] = fractions_at_times

        df = pd.DataFrame(data_for_table)
        st.dataframe(df.style.format("{:.2%}"))
