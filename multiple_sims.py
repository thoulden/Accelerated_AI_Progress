import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run():
    st.header("Multiple Simulations")

    run_sims = st.sidebar.button("Run Simulations")
    
    # User inputs for simulation setup
    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=10000, value=1000, step=100)
    simulation_duration = st.sidebar.number_input("Simulation Duration (years)", min_value=1, max_value=100, value=10)
    dt = 1.0 / 12.0  # 1 month time step
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
    

    if run_sims:
        # Parameter sampling function
        def sample_parameters():
            initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high)))
            initial_doubling_time = 3 / initial_boost

            r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high)))
            limit_years = np.random.uniform(ly_low, ly_high)
            lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high)))

            return r_initial, initial_doubling_time, limit_years, lambda_factor

        def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration):
            # Convert limit_years into the ceiling
            ceiling = 256 ** limit_years

            r = r_initial
            size = 1.0  # Starting size
            times = [0.0]
            sizes = [size]
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings if total_doublings > 0 else 0

            max_time = simulation_duration
            doubling_time = initial_doubling_time
            current_time = 0.0

            while size < ceiling and r > 0 and current_time < max_time:
                g_prime = np.log(2) / doubling_time
                size *= np.exp(g_prime * dt)
                current_time += dt
                if current_time > max_time:
                    break
                times.append(current_time)
                sizes.append(size)

                fraction_of_doubling = (g_prime * dt) / np.log(2)
                r -= k * fraction_of_doubling
                if r < 0:
                    r = 0
                if r > 0:
                    doubling_time *= 2 ** (lambda_factor * (1/r - 1))

            return np.array(times), np.array(sizes)

        # Run all simulations and store sizes
        progress = st.progress(0)  # Progress bar
        sim_sizes_list = []
        sim_times = None

        for i in range(num_sims):
            r_initial, initial_doubling_time, limit_years, lambda_factor = sample_parameters()
            times_array, sizes_array = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration)
            if sim_times is None:
                sim_times = times_array
            else:
                # Ensure all arrays match shortest length
                min_len = min(len(sim_times), len(times_array))
                sim_times = sim_times[:min_len]
                sim_sizes_list = [s[:min_len] for s in sim_sizes_list]
                sizes_array = sizes_array[:min_len]

            sim_sizes_list.append(sizes_array)
            progress.progress((i+1)/num_sims)

        # Convert to array
        sim_sizes_mat = np.array(sim_sizes_list) # (num_sims, T)
        # times shape (T,)

        # Compute growth rates
        # Assume uniform dt for simplicity
        growth_times = sim_times[1:]
        dt_years = growth_times[1] - growth_times[0] if len(growth_times) > 1 else 1.0

        sim_growth_rates = []
        for i in range(num_sims):
            sizes_arr = sim_sizes_mat[i]
            gr = (np.log(sizes_arr[1:]) - np.log(sizes_arr[:-1])) / dt_years
            sim_growth_rates.append(gr)
        sim_growth_rates = np.array(sim_growth_rates) # (num_sims, T-1)

        # For each multiple, compute fraction of sims exceeding multiple*g
        fractions_dict = {}
        for m in multiples:
            threshold = m * g
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
        report_times = [1, 2, 5, simulation_duration]
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

        # Format the table
        # We do not want years as percent, only fraction columns.
        # Fraction columns are all except "Time (years)"
        fraction_cols = [c for c in df.columns if c != 'Time (years)']
        format_dict = {col: "{:.2%}" for col in fraction_cols}

        # Hide row numbers and only format fraction columns as percent
       styled_df = df.style.format(format_dict)
        st.write(styled_df)

