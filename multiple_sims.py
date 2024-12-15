import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run():
    st.header("Multiple Simulations")

    # User inputs for simulation setup
    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=10000, value=1000, step=100)
    simulation_duration = st.sidebar.number_input("Simulation Duration (years)", min_value=1, max_value=100, value=10)
    dt = 1.0 / 12.0  # 1 month time step
    g = 2.77

    # Allow user to specify multiples
    multiples_input = st.sidebar.text_input("Enter speed-up multiples of g (comma-separated)", value="3,10,30")
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]

    # Allow user to specify time windows
    st.sidebar.markdown("### Time Window for Growth Conditions")
    time_windows = st.sidebar.text_input("Enter time windows (years, comma-separated)", value="1,3,5,10")
    time_windows = [float(t.strip()) for t in time_windows.split(",")]

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
        # Function to calculate summary statistics
        def calculate_summary_statistics_binary(times, conditions):
            results = {condition: 'no' for condition in conditions}

            for time_period, speed_up_factor in conditions:
                baseline_doublings = (time_period / 12) * 8  # 8 doublings per year
                required_doublings = int(baseline_doublings * speed_up_factor)

                for i in range(len(times) - required_doublings):
                    time_span = times[i + required_doublings] - times[i]
                    if time_span < time_period * 12:  # Convert years to months
                        results[(time_period, speed_up_factor)] = 'yes'
                        break
            return results

        # Parameter sampling function
        def sample_parameters():
            initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high)))
            initial_doubling_time = 3 / initial_boost

            r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high)))
            limit_years = np.random.uniform(ly_low, ly_high)
            lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high)))

            return r_initial, initial_doubling_time, limit_years, lambda_factor

        # Dynamic system function
        def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration):
            ceiling = 256 ** limit_years

            r = r_initial
            size = 1.0
            times = [0.0]
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings if total_doublings > 0 else 0

            max_time = simulation_duration
            doubling_time = initial_doubling_time
            current_time = 0.0

            while size < ceiling and r > 0 and current_time < max_time:
                g_prime = np.log(2) / doubling_time
                size *= np.exp(g_prime * dt)
                current_time += dt
                times.append(current_time)

                fraction_of_doubling = (g_prime * dt) / np.log(2)
                r -= k * fraction_of_doubling
                if r > 0:
                    doubling_time *= 2 ** (lambda_factor * (1 / r - 1))

            return np.array(times)

        # Conditions to check
        conditions = [(t, m) for t in time_windows for m in multiples]

        # Initialize progress bar
        progress = st.progress(0)

        # Run simulations and collect results
        summary_stats = []
        for i in range(num_sims):
            r_initial, initial_doubling_time, limit_years, lambda_factor = sample_parameters()
            times_array = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration)
            stats = calculate_summary_statistics_binary(times_array, conditions)
            summary_stats.append(stats)
            progress.progress((i + 1) / num_sims)

        # Aggregate results
        results_count = {condition: 0 for condition in conditions}
        for stats in summary_stats:
            for condition, result in stats.items():
                if result == 'yes':
                    results_count[condition] += 1

        results_fraction = {k: v / num_sims for k, v in results_count.items()}

        # Create table
        table_data = []
        for time_window in time_windows:
            row = {"Time Window (Years)": time_window}
            for multiple in multiples:
                key = (time_window, multiple)
                row[f"{multiple}x g"] = f"{results_fraction.get(key, 0):.2%}"
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.write("### Summary Table")
        styled_df = df.style.format(format_dict)
        st.write(styled_df)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        for multiple in multiples:
            fractions = [results_fraction.get((tw, multiple), 0) for tw in time_windows]
            ax.plot(time_windows, fractions, label=f"{multiple}x g", marker='o')

        ax.set_xlabel("Time Window (Years)")
        ax.set_ylabel("Fraction of Simulations Meeting Condition")
        ax.set_title("Fraction of Simulations with Growth Rate Exceeding Multiples of g")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)

