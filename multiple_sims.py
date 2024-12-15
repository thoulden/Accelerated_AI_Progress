import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

def run():

    run_sims = st.sidebar.button("Run Simulations")
    
    # User inputs for simulation setup
    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=10000, value=1000, step=100)
    simulation_duration = 4
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
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings if total_doublings > 0 else 0

            max_time = simulation_duration
            doubling_time = initial_doubling_time
            current_time = 0.0

            while current_time < max_time:
                g_prime = np.log(2) / doubling_time
                size *= np.exp(g_prime * dt)
                current_time += dt
                if current_time > max_time:
                    break
                times.append(current_time)

                fraction_of_doubling = (g_prime * dt) / np.log(2)
                r -= k * fraction_of_doubling
                if r < 0:
                    r = 0
                if r > 0:
                    doubling_time *= 2 ** (lambda_factor * (1/r - 1))

            return np.array(times)

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

        # Define periods and speed-up factors
        time_periods = [0.083, 0.25, 1.0, 3.0]  # 1 month, 3 months, 1 year, 3 years
        conditions = list(product(time_periods, multiples))

        # Run simulations and collect results
        progress = st.progress(0)
        times_matrix = []

        for i in range(num_sims):
            r_initial, initial_doubling_time, limit_years, lambda_factor = sample_parameters()
            times_array = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, lambda_factor, dt, simulation_duration)
            times_matrix.append(times_array)
            progress.progress((i + 1) / num_sims)

        # Aggregate results
        batch_summary_statistics = {condition: 0 for condition in conditions}

        for times in times_matrix:
            simulation_statistics = calculate_summary_statistics_binary(times, conditions)

            for condition in conditions:
                if simulation_statistics[condition] == 'yes':
                    batch_summary_statistics[condition] += 1

        batch_summary_fractions = {condition: count / num_sims for condition, count in batch_summary_statistics.items()}

        # Create a table
        table_data = []
        for time_period in time_periods:
            row = {"Time Window (Years)": time_period}
            for multiple in multiples:
                key = (time_period, multiple)
                row[f"{multiple}x g"] = f"{batch_summary_fractions.get(key, 0):.2%}"
            table_data.append(row)

        df = pd.DataFrame(table_data)

        # Format and display the table
        # Ensure no invalid data in fraction columns
           st.dataframe(df)  # Display plain DataFrame without styling


