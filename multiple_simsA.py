import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

def sample_parameters_batch(n_samples, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, compute_growth):
    """
    Sample n_samples sets of parameters in a vectorized manner, ensuring consistency in dimensions.
    Returns:
        A NumPy array of shape (n_samples, 8) containing sampled parameters.
    """
    # Log-uniform distributions
    initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high), size=n_samples))
    r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high), size=n_samples))
    limit_years = np.random.uniform(ly_low, ly_high, size=n_samples)
    lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high), size=n_samples))

    # Scalars or fixed values
    factor_increase = 2 if compute_growth else 2 # this can be adjusted so that the model is run in smaller steps than doublings. This would requre changes to the calculate binary statistics function. 
    compute_growth_monthly_rate = np.log(2) / 3  # Fixed scalar for monthly compute growth rate
    implied_month_growth_rate = np.log(2) / 3  # Fixed scalar for implied monthly growth rate
    time_takes_to_factor_increase = np.log(factor_increase) / implied_month_growth_rate
    if compute_growth:
        # denominator is 1.1, but produce an array of shape (n_samples,)  assumes that if we have compute growth the starting boost is 1 + 0.1
        denominator = np.full(n_samples, 1.1)
    else:
        # denominator is initial_boost, which is already shape (n_samples,)
        denominator = initial_boost
    initial_factor_increase_time = time_takes_to_factor_increase / denominator
    #initial_factor_increase_time = time_takes_to_factor_increase / (1.1 if compute_growth else initial_boost) 
    #initial_factor_increase_time = time_takes_to_factor_increase / (1 + (0.1 if compute_growth else initial_boost))  # Matches each initial_boost

    # Variables dependent on initial_boost
    f_0 = np.full(n_samples, 0.1) if compute_growth else initial_boost  # Matches each draw of initial_boost
    f_max = initial_boost  # f_max depends on initial_boost

    # Stack the parameters into a consistent array
    return np.column_stack((
        r_initial,                     # 1
        factor_increase * np.ones(n_samples),  # 2  <-- new column
        initial_factor_increase_time,  # 3
        limit_years,                   # 4
        np.full(n_samples, compute_growth_monthly_rate),  # 5
        f_0,                           # 6
        f_max,                         # 7
        lambda_factor                  # 8
    ))

def dynamic_system_with_lambda(size_adjustment, r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, lambda_factor, retraining_cost, max_time_months=48):
    ceiling = 256 ** limit_years
    size = 1.0
    r = r_initial
    f = f_0

    times, sizes, rs, compute_sizes, f_values = [0], [size], [r], [1], [f]
    time_elapsed = 0
    k = r_initial / (np.log(ceiling) / np.log(factor_increase))

    while time_elapsed < max_time_months and size < ceiling and r > 0:
        f_old = f
        time_elapsed += initial_factor_increase_time
        size *= factor_increase
        compute_size = compute_sizes[-1] * np.exp(compute_growth_monthly_rate * initial_factor_increase_time)

        if compute_size < 4096:
            f = f_0 + (f_max - f_0) * (np.log(compute_size) / np.log(4096))
        else:
            f = f_max

        r -= k
        times.append(time_elapsed)
        sizes.append(size)
        rs.append(r)
        compute_sizes.append(compute_size)
        f_values.append(f)
        if r > 0:
            accel_factor = ((lambda_factor * ((1 / r) - 1))/(abs(lambda_factor * ((1 / r) - 1) + 1))) if retraining_cost else lambda_factor * (1 / r - 1)
            #initial_factor_increase_time *= (factor_increase ** accel_factor) / ((1 + f) / (1 + f_old))
            if size_adjustment:
                    initial_factor_increase_time *= ((factor_increase ** accel_factor) / ((1 + f) / (1 + f_old)))* (size ** (1/r - 1/rs[-2])) #TH mehtod with size adjustment
            else:         
                    initial_factor_increase_time *= ((factor_increase ** accel_factor) / ((1 + f) / (1 + f_old))) #TD's method
    return times, sizes, rs, compute_sizes, f_values

def calculate_summary_statistics_binary(times, conditions):
    results = {condition: 'no' for condition in conditions}

    for time_period, speed_up_factor in conditions:
        baseline_doublings = (time_period / 12) * 8
        required_doublings = int(baseline_doublings * speed_up_factor)

        for i in range(len(times) - required_doublings):
            time_span = times[i + required_doublings] - times[i]
            if time_span < time_period:
                results[(time_period, speed_up_factor)] = 'yes'
                break

    return results

def run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, retraining_cost, compute_growth, size_adjustment):
    params_batch = sample_parameters_batch(num_sims, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, compute_growth)
    times_matrix = []
    progress = st.progress(0)

    for i, params in enumerate(params_batch):
        r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, lambda_factor = params
        times, _, _, _, _ = dynamic_system_with_lambda(
            size_adjustment, r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, lambda_factor, retraining_cost)
        times_matrix.append(times)
        progress.progress((i + 1) / num_sims)

    batch_summary = {condition: 0 for condition in conditions}
    for times in times_matrix:
        stats = calculate_summary_statistics_binary(times, conditions)
        for condition in conditions:
            if stats[condition] == 'yes':
                batch_summary[condition] += 1

    return {condition: count / num_sims for condition, count in batch_summary.items()}

def to_markdown_table(df):
        """
         Convert a small pandas DataFrame to a markdown table (no index).
        """
        df = df.reset_index(drop=True)
        header = "| " + " | ".join(df.columns) + " |\n"
        separator = "| " + " | ".join("---" for _ in df.columns) + " |\n"

        rows = []
        for row_tuple in df.itertuples(index=False):
            row_str = "| " + " | ".join(str(x) for x in row_tuple) + " |"
            rows.append(row_str)

        return header + separator + "\n".join(rows)
    
def run():
    run_button = st.sidebar.button("Run Simulations")

    st.sidebar.markdown("### Key Parameter Sampling Bounds")
    ib_low = st.sidebar.number_input(r"Boost ($f$); lower bound", min_value=0.1, value=2.0)
    ib_high = st.sidebar.number_input(r"Boost ($f$); upper bound)", min_value=ib_low, value=32.0)
    r_low = st.sidebar.number_input(r"Initial Productivity ($r_0$); lower bound", min_value=0.01, value=0.4)
    r_high = st.sidebar.number_input(r"Initial Productivity ($r_0$); upper bound", min_value=r_low, value=3.6)
    ly_low = st.sidebar.number_input("Years to Ceiling; lower bound", min_value=1.0, value=7.0)
    ly_high = st.sidebar.number_input("Years to Ceiling; upper bound", min_value=ly_low, value=14.0)
    lf_low = st.sidebar.number_input(r"Parallelizability ($\lambda$); lower bound", min_value=0.01, value=0.2)
    lf_high = st.sidebar.number_input(r"Parallelizability ($\lambda$); upper bound", min_value=lf_low, value=0.8)

    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=30000, value=1000, step=100)
    multiples_input = st.sidebar.text_input("Multiples (comma-separated)", value="3,10,30")
    multiples_input = st.sidebar.text_input("Growth Multiples (comma-separated)", value="3,10,30")

    retraining_cost = st.sidebar.checkbox("Retraining Cost")
    # size_adjustment = st.sidebar.checkbox("size_adjustment") # old code for size adjustment to match SEG results
    size_adjustment = 'false'
    compute_growth = st.sidebar.checkbox("Compute Growth")
    
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]
    conditions = list(product([1, 4, 12, 36], multiples))

    if run_button:
        results = run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, retraining_cost, compute_growth, size_adjustment)

        data = []
        for time_period in sorted(set(c[0] for c in results.keys())):
            row = {"Time Period (Months)": time_period}
            for multiple in sorted(set(c[1] for c in results.keys())):
                row[f"{multiple}x faster"] = results.get((time_period, multiple), 0)
            data.append(row)
        # Convert 'data' (list of dicts) to a DataFrame
        df = pd.DataFrame(data)

        # Example usage:
        md_table = to_markdown_table(df)
        st.write("###### What is the probability AI progress is X times faster for N months?")
        st.markdown(md_table)
    else:
        st.write("Press 'Run Simulation' to view results.")
    
    

    

if __name__ == "__main__":
    run()
