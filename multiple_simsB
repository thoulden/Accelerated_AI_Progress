import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

def sample_parameters_batch(n_samples, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, compute_growth, software_contribution):
    """
    Sample n_samples sets of parameters in a vectorized manner, ensuring consistency in dimensions.
    Returns:
        A NumPy array of shape (n_samples, 9) containing sampled parameters.
    """
    # Log-uniform distributions
    initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high), size=n_samples))
    r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high), size=n_samples))
    limit_years = np.random.uniform(ly_low, ly_high, size=n_samples)
    lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high), size=n_samples))

    # Scalars or fixed values
    factor_increase = 2 if compute_growth else 2 # this can be adjusted so that the model is run in smaller steps than doublings. This would requre changes to the calculate binary statistics function. 
    compute_growth_monthly_rate = np.log(2) / 5  # Fixed scalar for monthly compute growth rate, compute doubles every 5 months
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
        lambda_factor,                 # 8
        np.full(n_samples, software_contribution)  # 9 - ADD software_contribution
    ))

def dynamic_system_with_lambda(r_initial, factor_increase, initial_factor_increase_time, limit_years, 
                               compute_growth_monthly_rate, f_0, f_max, lambda_factor, software_contribution,
                               retraining_cost, constant_r, max_time_months=48):
    # Updated ceiling calculation using software_contribution
    ceiling = (2**(4/software_contribution)) ** limit_years
    size = 1.0
    r = r_initial
    f = f_0

    times, sizes, rs, compute_sizes, f_values = [0], [size], [r], [1], [f]
    time_elapsed = 0
    if constant_r:
        k = 0
    else:
        k = r_initial / (np.log(ceiling) / np.log(factor_increase))

    while time_elapsed < max_time_months and size < ceiling and r > 0:
        f_old = f
        time_elapsed += initial_factor_increase_time
        size *= factor_increase
        compute_size = compute_sizes[-1] * np.exp(compute_growth_monthly_rate * initial_factor_increase_time)

        if compute_size < 4096:
            f_growth_rate = np.log(f_max/f_0)/(5*12) #reaches ceiling in 5 years
            f =  f_0*np.exp(time_elapsed*f_growth_rate) #ensures exponential growth in f; options below are for old linear f approach       
            #f = f_0 + (f_max - f_0) * (compute_size / 4096) #for exponential f growth
            #f = f_0 + (f_max - f_0) * (np.log(compute_size) / np.log(4096)) # 12 doublings, for linear f growth
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
            initial_factor_increase_time *= ((factor_increase ** accel_factor) / ((1 + f) / (1 + f_old))) 
    return times, sizes, rs, compute_sizes, f_values

def calculate_summary_statistics_binary(times, conditions, software_contribution):
    results = {condition: 'no' for condition in conditions}
    
    # Calculate the annual number of doublings based on software_contribution
    annual_doublings = 4 / software_contribution

    for time_period, speed_up_factor in conditions:
        baseline_doublings = (time_period / 12) * annual_doublings
        required_doublings = int(baseline_doublings * speed_up_factor)

        for i in range(len(times) - required_doublings):
            time_span = times[i + required_doublings] - times[i]
            if time_span < time_period:
                results[(time_period, speed_up_factor)] = 'yes'
                break

    return results

def run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, 
                   retraining_cost, compute_growth, constant_r, software_contribution):
    params_batch = sample_parameters_batch(num_sims, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, 
                                          compute_growth, software_contribution)
    times_matrix = []
    sizes_matrix = []
    params_list = []

    progress = st.progress(0)

    for i, params in enumerate(params_batch):
        r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, lambda_factor, software_contribution = params
        times, sizes, rs, compute_sizes, f_values = dynamic_system_with_lambda(
         r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, 
         f_0, f_max, lambda_factor, software_contribution, retraining_cost, constant_r)
        times_matrix.append(times)
        sizes_matrix.append(sizes)
        params_list.append({
            "r_initial": r_initial,
            "factor_increase": factor_increase,
            "initial_factor_increase_time": initial_factor_increase_time,
            "limit_years": limit_years,
            "compute_growth_monthly_rate": compute_growth_monthly_rate,
            "f_0": f_0,
            "f_max": f_max,
            "lambda_factor": lambda_factor,
            "software_contribution": software_contribution
        })
        progress.progress((i + 1) / num_sims)

    batch_summary = {condition: 0 for condition in conditions}
    for times in times_matrix:
        stats = calculate_summary_statistics_binary(times, conditions, software_contribution)
        for condition in conditions:
            if stats[condition] == 'yes':
                batch_summary[condition] += 1
    # Probability of condition = count / total number of simulations
    probabilities = {
        condition: count / num_sims
        for condition, count in batch_summary.items()
    }
    return probabilities, times_matrix, sizes_matrix, params_list

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
    ib_low = st.sidebar.number_input(r"Initial speed-up ($f$); lower bound", min_value=0.1, value=2.0, help="After ASARA is deployed, how much faster is software progress compared to the recent pace of software progress?")
    ib_high = st.sidebar.number_input(r"Initial speed-up ($f$); upper bound", min_value=ib_low, value=32.0)
    r_low = st.sidebar.number_input(r"Returns to Software R&D ($r$); lower bound", min_value=0.01, value=0.4, help="Each time cumulative inputs to software R&D double, how many times does software double? (Any improvement with the same benefits as running 2x more parallel copies of the same AI corresponds to a doubling of software.)")
    r_high = st.sidebar.number_input(r"Returns to Software R&D ($r$); upper bound", min_value=r_low, value=3.6)
    ly_low = st.sidebar.number_input("Distance to effective limits on software; lower bound", min_value=1.0, value=6.0, help="When ASARA is first developed, how far is AI software from effective limits? (Measured in units of "years of AI progress at the recent rate of progress".)")
    ly_high = st.sidebar.number_input("Distance to effective limits on software; upper bound", min_value=ly_low, value=16.0)
    lf_low = st.sidebar.number_input(r"Diminishing returns to parallel labour ($p$); lower bound", min_value=0.01, value=0.15, help="If you instantaneously doubled the amount of parallel cognitive labour directed towards software R&D, how many times would the pace of software progress double?")
    lf_high = st.sidebar.number_input(r"Diminishing returns to parallel labour ($p$); upper bound", min_value=lf_low, value=0.6)
    software_contribution_param = st.sidebar.number_input(r"Fraction of total AI progress that is due to better software (rather than more compute)?", min_value=0.01, max_value=0.99,
                                        value=0.5, step=0.01,
                                        help="A larger fraction means that the software progress modelled contributes more to the overall AI progress.")

    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=30000, value=1000, step=100)
    multiples_input = st.sidebar.text_input("Growth Multiples (comma-separated)", value="3,10,30", help="These are the comparisson multiples (of the current growth rate) that are reported in the results.")

    compute_growth = st.sidebar.checkbox("Gradual Boost", help = "The initial speed-up from ASARA ramps up gradually over 5 years.")
    retraining_cost = st.sidebar.checkbox("Retraining Cost", help = "Reduce the degree of acceleration as some software efficiency gains are spent making training happen more quickly.")
    constant_r = st.sidebar.checkbox("Constant Diminishing Returns", help = "Assumes that $r$ is fixed at its initial value over time.")
    
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]
    conditions = list(product([1, 4, 12, 36], multiples))

    if run_button:
        probabilities, times_matrix, sizes_matrix, params_list = run_simulations(
            num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, 
            retraining_cost, compute_growth, constant_r, software_contribution_param)

        data = []
        for time_period in sorted(set(c[0] for c in probabilities.keys())):
            row = {"Time Period (Months)": time_period}
            for multiple in sorted(set(c[1] for c in probabilities.keys())):
                row[f"{multiple}x faster"] = probabilities.get((time_period, multiple), 0)
            data.append(row)
        # Convert 'data' (list of dicts) to a DataFrame
        df = pd.DataFrame(data)

        # Example usage:
        md_table = to_markdown_table(df)
        st.write("###### What is the probability AI progress is X times faster for N months?")
        st.write("(More precisely, what is the probability that there is an N month period where the average pace of AI software progress is X times faster than the recent pace of overall AI progress?)")
        st.markdown(md_table)

        #Create DataFrame for times & sizes
        simulation_results = []
        for i, (times, sizes) in enumerate(zip(times_matrix, sizes_matrix)):
            # Retrieve this simulation's parameters
            sim_params = params_list[i]
            # For each timestep in this simulation, add a row
            for t, s in zip(times, sizes):
                simulation_results.append({
                    "Simulation": i + 1,
                    "r_initial": sim_params["r_initial"],
                    # "factor_increase": sim_params["factor_increase"],
                    "initial_factor_increase_time": sim_params["initial_factor_increase_time"],
                    "limit_years": sim_params["limit_years"],
                    # "compute_growth_monthly_rate": sim_params["compute_growth_monthly_rate"],
                    "f_0": sim_params["f_0"],
                    "f_max": sim_params["f_max"],
                    "lambda_factor": sim_params["lambda_factor"],
                    "software_contribution": sim_params["software_contribution"],
                    "Time (Months)": t,
                    "Size": s
                })

        df_results = pd.DataFrame(simulation_results)

        # Convert to CSV & create Download Button
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            label="Download Simulation Results (CSV)",
            data=csv_data,
            file_name="simulation_results.csv",
            mime="text/csv",
        )
        st.write("Download recommended for < 2000 simulations only.")
    else:
        st.write("Press 'Run Simulation' to view results.")
    
    

    

if __name__ == "__main__":
    run()
