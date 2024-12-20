import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

def run():

    run_sims = st.sidebar.button("Run Simulations")

    # Allow user to specify parameter sampling bounds
    st.sidebar.markdown("### Key Parameter Sampling Bounds")
    
    
    st.sidebar.markdown("#### Acceleration factor (f, log-uniform)")
    
    # Option to compute growth
    st.sidebar.markdown("#### Boost (f, log-uniform)")
    ib_low = st.sidebar.number_input("low bound", min_value=0.1, value=2.0)
    ib_high = st.sidebar.number_input("high bound", min_value=ib_low, value=32.0) #will say the difference between compute growth or not is slow progress up to chocie

    st.sidebar.markdown("#### Initial Research Productivity (r₀, log-uniform)")
    r_low = st.sidebar.number_input("low bound", min_value=0.01, value=0.4)
    r_high = st.sidebar.number_input("high bound", min_value=r_low, value=3.6)

    st.sidebar.markdown("#### Years Till Ceiling (log-uniform)")
    ly_low = st.sidebar.number_input("low bound", min_value=1.0, value=7.0)
    ly_high = st.sidebar.number_input("high bound", min_value=ly_low, value=14.0)

    st.sidebar.markdown("#### Parallelizability (λ, log-uniform)")
    lf_low = st.sidebar.number_input("low bound", min_value=0.01, value=0.2)
    lf_high = st.sidebar.number_input("high bound", min_value=lf_low, value=0.8)
    
    st.sidebar.markdown("### Additional Choices")
    # Checkbox for retraining cost
    retraining_cost = st.sidebar.checkbox('Retraining Cost')
    compute_growth = st.sidebar.checkbox('Compute Growth')
    # User inputs for simulation setup
    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=30000, value=1000, step=100)
    simulation_duration = 4

    # Allow user to specify multiples
    multiples_input = st.sidebar.text_input("Test growth rates exceeding how many times usual growth? (comma-separated)", value="3,10,30")
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]

    if run_sims:
        # Parameter sampling function

        def sample_parameters():
            """
            Sample initial parameters from uniform and log-uniform distributions.

            Returns:
            r_initial: The initial value of r (diminishing returns).
            initial_doubling_time: Initial doubling time in months.
            limit_years: The limit expressed as years of progress at recent rates.
            lambda_factor: The lambda factor for adjusting doubling time.
            """

            # Sample r from a log-uniform distribution (range: 0.4 to 3.6)
            r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high)))

            # Sample limit_years uniformly from 7 to 14
            limit_years = np.random.uniform(ly_low, ly_high)

            # Sample lambda factor from log-uniform distribution (range: 0.2 to 0.8)
            lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high)))

            # Have to interpret boost differently depending on compute growing
            initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high)))

            if compute_growth:
                factor_increase = 1.1  # Set the desired factor increase (e.g., 1.1 for 10% increases)
                f_0 = 0.1 # ensures that boost starts low and goes to 'inital boost' which we can interpret as max boost here
            else: 
                factor_increase = 2 # when not doing compute growing just use doublings
                f_0 = initial_boost # ensures that boost starts high
            f_max = initial_boost
            compute_size_start = 1
            compute_max = 4096
            compute_doubling_time = 3
            compute_growth_monthly_rate = np.log(2) / compute_doubling_time
            doubling_time_starting = 3 #months
            implied_month_growth_rate = np.log(2)/doubling_time_starting
            time_takes_to_factor_increase = np.log(factor_increase)/implied_month_growth_rate
            initial_factor_increase_time = time_takes_to_factor_increase / (1+f_0)

            return r_initial, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, compute_size_start, factor_increase, lambda_factor)

            def dynamic_system_with_lambda(r_initial, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, compute_size_start, compute_max, factor_increase, lambda_factor=0.5, max_time_months=48):
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
            factor_increase_time = initial_factor_increase_time
            size = 1.0
            compute_size = compute_size_start

            # Lists to store outputs
            times = [0]
            sizes = [size]
            rs = [r]
            compute_sizes = [compute_size]
            f_values = [f_0]
            f=f_0
            # Calculate total factor increasings
            total_factor_increasings = np.log(ceiling) / np.log(factor_increase)
            k = r_initial / total_factor_increasings
            time_elapsed = 0
            while time_elapsed < max_time_months and size < ceiling and r > 0:
                # Store previous f for updates
                f_old = f
                
                time_step = factor_increase_time
                time_elapsed += time_step
                times.append(time_elapsed)
                size *= factor_increase
                sizes.append(size)
                r -= k
                rs.append(r)

                # Update compute size
                compute_size = compute_size_start * np.exp(compute_growth_monthly_rate * time_elapsed)
                compute_sizes.append(compute_size)

                # Update acceleration factor f
                if compute_size < compute_max:
                    f = f_0 + (f_max - f_0) * (np.log(compute_size / compute_size_start) / np.log(compute_max / compute_size_start))
                else:
                    f = f_max
                f_values.append(f)

                # Set factor increasing factor
                if r > 0:
                    if retraining_cost:
                        accel_factor = ((lambda_factor * ((1 / r) - 1))/(abs(lambda_factor * ((1 / r) - 1) + 1)))
                    else: 
                        accel_factor = (lambda_factor * (1 / r - 1)) 
                    factor_increase_time *= (factor_increase ** accel_factor) / ((1 + f) / (1 + f_old))
            return times, sizes, rs, ceiling, compute_sizes, f_values

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
            initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high), size=n_samples))
            initial_doubling_time = 3 / initial_boost  # Initial doubling time

            # Log-uniform distribution for r (0.4 to 3.6)
            r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high), size=n_samples))

            # Uniform distribution for limit_years (7 to 14)
            limit_years = np.random.uniform(ly_low, ly_high, size=n_samples)

            # Log-uniform distribution for lambda factor (0.2 to 0.8)
            lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high), size=n_samples))
        
            return np.column_stack((r_initial, initial_doubling_time, limit_years, lambda_factor))

        # Run sims

        # Define periods (in months) and speed-up factors
        time_periods = [1, 4, 12, 36]  # Corresponding to 4 months, 12 months, and 36 months #OLD
        #speed_up_factors = [3, 6, 10, 30]  # Corresponding to 3X, 6X 10X, and 30X speedups #OLD
        speed_up_factors = multiples #incorporating user choice
        # Create a list of all combinations of time periods and speed-up factors
        from itertools import product
        conditions = list(product(time_periods, speed_up_factors))
        
        # Number of simulations
        n_sims = num_sims

        # Run  simulations with independently sampled parameters
        times_matrix = []

        # Progress bar
        progress = st.progress(0)
        for i in range(n_sims):
            # Sample a parameter set
            r, initial_doubling_time, limit_years, lambda_factor = sample_parameters_batch(1)[0]
        
            # Run the simulation
            times, _, _ = dynamic_system_with_lambda(
                r, initial_doubling_time, limit_years, lambda_factor=lambda_factor
            )

            # Save the times
            times_matrix.append(times)

            # Update progress bar
            progress.progress((i + 1) / n_sims)  

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
        #st.write("Fraction of simulations meeting each condition:")
        #for condition, fraction in batch_summary_fractions.items():
            #st.write(f"Condition {condition}: {fraction:.2%}")

        # Convert results into a DataFrame format
        time_periods = sorted(set(condition[0] for condition in batch_summary_fractions.keys()))
        multiples = sorted(set(condition[1] for condition in batch_summary_fractions.keys()))

        # Create a DataFrame with rows as time periods and columns as multiples
        data = []
        for time_period in time_periods:
            row = {"Time Period (Months)": time_period}
            for multiple in multiples:
                row[f"{multiple}x faster"] = batch_summary_fractions.get((time_period, multiple), 0)
            data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data)
    
               # Format the DataFrame
        df = df.sort_values(by="Time Period (Months)").reset_index(drop=True)  # Sort by time period
        fraction_cols = [f"{m}x faster" for m in multiples]
        format_dict = {col: "{:.2%}" for col in fraction_cols}

        # Display the table
        # Format the table without row numbers
        formatted_df = df.style.format(format_dict)

        # Display using st.table (removes row numbers)
        #st.write("###### What is the probability AI progress is X times faster for N months?")
        #st.table(formatted_df.data)

        # Convert the DataFrame to HTML and remove the row index
        html_table = formatted_df.hide(axis='index').to_html()

        # Use st.markdown to display the table without row numbers
        st.write("###### What is the probability AI progress is X times faster for N months?")
        st.markdown(html_table, unsafe_allow_html=True)


    else:
        st.write("Press 'Run Simulation' to view results.")


