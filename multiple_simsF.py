import streamlit as st
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

def sample_parameters_batch(n_samples, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, software_contribution_param, compute_growth):
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
    factor_increase = 2  # Using doublings
    compute_growth_monthly_rate = np.log(2) / 5  # Fixed scalar for monthly compute growth rate, compute doubles every 5 months
    implied_month_growth_rate = np.log(2) / 3  # Fixed scalar for implied monthly growth rate
    time_takes_to_factor_increase = np.log(factor_increase) / implied_month_growth_rate
    if compute_growth:
        # denominator is 1.1, but produce an array of shape (n_samples,)
        denominator = np.full(n_samples, 1.1)
    else:
        # denominator is initial_boost, which is already shape (n_samples,)
        denominator = initial_boost
    initial_factor_increase_time = time_takes_to_factor_increase / denominator

    # Variables dependent on initial_boost
    f_0 = np.full(n_samples, 0.1) if compute_growth else initial_boost
    f_max = initial_boost  # f_max depends on initial_boost

    # Stack the parameters into a consistent array
    return np.column_stack((
        r_initial,                     # 1
        factor_increase * np.ones(n_samples),  # 2
        initial_factor_increase_time,  # 3
        limit_years,                   # 4
        np.full(n_samples, compute_growth_monthly_rate),  # 5
        f_0,                           # 6
        f_max,                         # 7
        lambda_factor,                 # 8
        np.full(n_samples, software_contribution_param)  # 9
    ))

def dynamic_system_with_lambda(r_initial, factor_increase, initial_factor_increase_time, limit_years, compute_growth_monthly_rate, f_0, f_max, lambda_factor, software_contribution_param, retraining_cost, constant_r, max_time_months=48):
    # Fixed exponentiation syntax
    ceiling = (2**(4/software_contribution_param)) ** limit_years
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
            f_growth_rate = np.log(f_max/f_0)/(5*12)  # reaches ceiling in 5 years
            f = f_0*np.exp(time_elapsed*f_growth_rate)  # ensures exponential growth in f
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

def calculate_summary_statistics_binary(times, conditions, software_contribution_param):
    results = {condition: 'no' for condition in conditions}

    for time_period, speed_up_factor in conditions:
        baseline_doublings = (time_period / 12) * (4/software_contribution_param)
        required_doublings = int(baseline_doublings * speed_up_factor)

        for i in range(len(times) - required_doublings):
            time_span = times[i + required_doublings] - times[i]
            if time_span < time_period:
                results[(time_period, speed_up_factor)] = 'yes'
                break

    return results

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

def calculate_years_compressed_cdf(times_matrix, software_contribution_param, time_period_months=12, max_years=20, resolution=200):
    """
    Calculate the CDF for years of progress compressed into a given time period.
    For the plot, X-axis represents "years of progress to compress"
    """
    years_points = np.linspace(0.1, max_years, resolution)
    fractions = []
    
    for years_to_compress in years_points:
        # Calculate how much speed-up is needed to compress this many years into the time period
        # If we want to compress X years into time_period_months:
        # speed_up_factor = X / (time_period_months/12)
        speed_up_factor = years_to_compress / (time_period_months / 12)
        
        # Now use the same calculation as the table
        baseline_doublings = (time_period_months / 12) * (4/software_contribution_param)
        required_doublings = baseline_doublings * speed_up_factor
        
        # Use floor and ceiling to interpolate
        doublings_floor = int(np.floor(required_doublings))
        doublings_ceil = int(np.ceil(required_doublings))
        
        if doublings_floor == doublings_ceil:
            # Exact integer case
            success_count = 0
            for times in times_matrix:
                if doublings_floor < len(times):
                    achieved = False
                    for i in range(len(times) - doublings_floor):
                        time_span = times[i + doublings_floor] - times[i]
                        if time_span < time_period_months:
                            achieved = True
                            break
                    if achieved:
                        success_count += 1
            fraction = success_count / len(times_matrix)
        else:
            # Interpolate between floor and ceiling
            success_count_floor = 0
            for times in times_matrix:
                if doublings_floor < len(times):
                    achieved = False
                    for i in range(len(times) - doublings_floor):
                        time_span = times[i + doublings_floor] - times[i]
                        if time_span < time_period_months:
                            achieved = True
                            break
                    if achieved:
                        success_count_floor += 1
            
            success_count_ceil = 0
            for times in times_matrix:
                if doublings_ceil < len(times):
                    achieved = False
                    for i in range(len(times) - doublings_ceil):
                        time_span = times[i + doublings_ceil] - times[i]
                        if time_span < time_period_months:
                            achieved = True
                            break
                    if achieved:
                        success_count_ceil += 1
            
            # Interpolate
            weight = required_doublings - doublings_floor
            fraction = ((1 - weight) * success_count_floor + weight * success_count_ceil) / len(times_matrix)
        
        fractions.append(fraction)
    
    return years_points, fractions

def run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, software_contribution_param, retraining_cost, compute_growth, constant_r):
    params_batch = sample_parameters_batch(num_sims, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, software_contribution_param, compute_growth)
    times_matrix = []
    sizes_matrix = []
    params_list = []

    progress = st.progress(0)

    for i, params in enumerate(params_batch):
        # Unpack all 9 parameters
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
            "software_contribution_param": software_contribution
        })
        progress.progress((i + 1) / num_sims)

    batch_summary = {condition: 0 for condition in conditions}
    for times in times_matrix:
        stats = calculate_summary_statistics_binary(times, conditions, software_contribution_param)
        for condition in conditions:
            if stats[condition] == 'yes':
                batch_summary[condition] += 1
    # Probability of condition = count / total number of simulations
    probabilities = {
        condition: count / num_sims
        for condition, count in batch_summary.items()
    }
    return probabilities, times_matrix, sizes_matrix, params_list

def run():
    run_button = st.sidebar.button("Run Simulations")

    st.sidebar.markdown("### Key Parameter Sampling Bounds")
    ib_low = st.sidebar.number_input(r"Initial speed-up ($f$); lower bound", min_value=0.1, value=2.0, help="After ASARA is deployed, how much faster is software progress compared to the recent pace of software progress?")
    ib_high = st.sidebar.number_input(r"Initial speed-up ($f$); upper bound", min_value=ib_low, value=32.0)
    r_low = st.sidebar.number_input(r"Returns to Software R&D ($r$); lower bound", min_value=0.01, value=0.4, help="Each time cumulative inputs to software R&D double, how many times does software double? (Any improvement with the same benefits as running 2x more parallel copies of the same AI corresponds to a doubling of software.)")
    r_high = st.sidebar.number_input(r"Returns to Software R&D ($r$); upper bound", min_value=r_low, value=3.6)
    ly_low = st.sidebar.number_input("Distance to effective limits on software; lower bound", min_value=1.0, value=6.0, help="When ASARA is first developed, how far is AI software from effective limits? (Measured in units of years of AI progress at the recent rate of progress.)")
    ly_high = st.sidebar.number_input("Distance to effective limits on software; upper bound", min_value=ly_low, value=16.0)
    lf_low = st.sidebar.number_input(r"Diminishing returns to parallel labour ($p$); lower bound", min_value=0.01, value=0.15, help="If you instantaneously doubled the amount of parallel cognitive labour directed towards software R&D, how many times would the pace of software progress double?")
    lf_high = st.sidebar.number_input(r"Diminishing returns to parallel labour ($p$); upper bound", min_value=lf_low, value=0.6)

    st.sidebar.markdown("### Additional Simulation Options")
    software_contribution_param = st.sidebar.number_input(r"Fraction of total AI progress that is due to better software (rather than more compute)?", min_value=0.01, max_value=0.99,
                                        value=0.5, step=0.01,
                                        help="A larger fraction means that the software progress modelled contributes more to the overall AI progress.")

    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=30000, value=1000, step=100)

    compute_growth = st.sidebar.checkbox("Gradual Boost", help="The initial speed-up from ASARA ramps up gradually over 5 years.")
    retraining_cost = st.sidebar.checkbox("Retraining Cost", help="Reduce the degree of acceleration as some software efficiency gains are spent making training happen more quickly.")
    constant_r = st.sidebar.checkbox("Constant Diminishing Returns", help="Assumes that $r$ is fixed at its initial value over time.")
    
    # Fixed conditions for the specific table format
    conditions = [(12, 3), (4, 10), (12, 10), (4, 30)]

    if run_button:
        probabilities, times_matrix, sizes_matrix, params_list = run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, software_contribution_param, retraining_cost, compute_growth, constant_r)

        # Create the specific table format
        st.write("### Years of AI progress compressed into different time periods")
        
        # Debug: Show what the table conditions actually represent
        st.write("**Debug info:**")
        st.write("- '≥3 years' row, 1 year column: 3x speed for 12 months = 3 years compressed into 1 year")
        st.write("- '≥3 years' row, 4 months column: 10x speed for 4 months = 10×(4/12) = 3.33 years compressed into 4 months")
        st.write("- '≥10 years' row, 1 year column: 10x speed for 12 months = 10 years compressed into 1 year")
        st.write("- '≥10 years' row, 4 months column: 30x speed for 4 months = 30×(4/12) = 10 years compressed into 4 months")
        
        # Create the data for the table
        table_data = {
            "Years of progress": ["≥3 years", "≥10 years"],
            "Compressed into ≤1 year": [
                f"{probabilities.get((12, 3), 0)*100:.0f}%",
                f"{probabilities.get((12, 10), 0)*100:.0f}%"
            ],
            "Compressed into ≤4 months": [
                f"{probabilities.get((4, 10), 0)*100:.0f}%",
                f"{probabilities.get((4, 30), 0)*100:.0f}%"
            ]
        }
        
        df_table = pd.DataFrame(table_data)
        # Use markdown table to avoid row numbers
        md_table = to_markdown_table(df_table)
        st.markdown(md_table)
        
        # Calculate CDF data for both 1 year and 4 months
        years_points_12, fractions_12 = calculate_years_compressed_cdf(
            times_matrix, 
            software_contribution_param,
            time_period_months=12,
            max_years=20,
            resolution=200
        )
        
        years_points_4, fractions_4 = calculate_years_compressed_cdf(
            times_matrix, 
            software_contribution_param,
            time_period_months=4,
            max_years=20,
            resolution=200
        )
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both lines
        ax.plot(years_points_12, fractions_12, color='blue', linewidth=2, label='Compressed into 1 year')
        ax.plot(years_points_4, fractions_4, color='purple', linewidth=2, label='Compressed into 4 months')
        
        # Add vertical lines - dashed for 1 year, dotted for 4 months
        # For 3 years
        idx_3_12 = np.argmin(np.abs(years_points_12 - 3))
        idx_3_4 = np.argmin(np.abs(years_points_4 - 3))
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3 years (1 year)')
        ax.axvline(x=3, color='red', linestyle=':', alpha=0.5, label='3 years (4 months)')
        
        # For 10 years
        idx_10_12 = np.argmin(np.abs(years_points_12 - 10))
        idx_10_4 = np.argmin(np.abs(years_points_4 - 10))
        ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='10 years (1 year)')
        ax.axvline(x=10, color='green', linestyle=':', alpha=0.5, label='10 years (4 months)')
        
        ax.set_xlabel('Years Compressed', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Probability of Compressing X Years into One Year or Four Months', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Custom legend to avoid duplicates
        handles, labels = [], []
        # Main lines
        handles.append(plt.Line2D([0], [0], color='blue', linewidth=2))
        labels.append('Compressed into 1 year')
        handles.append(plt.Line2D([0], [0], color='purple', linewidth=2))
        labels.append('Compressed into 4 months')
        # Reference lines
        handles.append(plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7))
        labels.append('3 years reference')
        handles.append(plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.7))
        labels.append('10 years reference')
        
        ax.legend(handles, labels, fontsize=10)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)
        
        # Add percentage labels at key points for both curves
        # For 1 year compression - exact matches to table
        for years in [3, 10]:
            idx_12 = np.argmin(np.abs(years_points_12 - years))
            prob_12 = fractions_12[idx_12]
            ax.annotate(f'{prob_12*100:.0f}%', 
                       xy=(years, prob_12), 
                       xytext=(years+0.5, prob_12+0.05),
                       fontsize=10,
                       color='blue')
        
        # For 4 months compression - need to check at 3.33 and 10
        # 3.33 years corresponds to table's "≥3 years" row
        idx_4_3 = np.argmin(np.abs(years_points_4 - (10/3)))  # 10*(4/12) = 3.33
        prob_4_3 = fractions_4[idx_4_3]
        ax.annotate(f'{prob_4_3*100:.0f}%', 
                   xy=(3, prob_4_3),  # Place at x=3 for visual alignment
                   xytext=(3+0.5, prob_4_3-0.05),
                   fontsize=10,
                   color='purple')
        
        # 10 years for 4-month compression
        idx_4_10 = np.argmin(np.abs(years_points_4 - 10))
        prob_4_10 = fractions_4[idx_4_10]
        ax.annotate(f'{prob_4_10*100:.0f}%', 
                   xy=(10, prob_4_10), 
                   xytext=(10+0.5, prob_4_10-0.05),
                   fontsize=10,
                   color='purple')
        
        st.pyplot(fig)

        # Create DataFrame for times & sizes - only if simulations <= 2000
        if num_sims <= 2000:
            simulation_results = []
            for i, (times, sizes) in enumerate(zip(times_matrix, sizes_matrix)):
                # Retrieve this simulation's parameters
                sim_params = params_list[i]
                # For each timestep in this simulation, add a row
                for t, s in zip(times, sizes):
                    simulation_results.append({
                        "Simulation": i + 1,
                        "r_initial": sim_params["r_initial"],
                        "initial_factor_increase_time": sim_params["initial_factor_increase_time"],
                        "limit_years": sim_params["limit_years"],
                        "f_0": sim_params["f_0"],
                        "f_max": sim_params["f_max"],
                        "lambda_factor": sim_params["lambda_factor"],
                        "software_contribution": sim_params["software_contribution_param"],
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
        else:
            st.info(f"CSV download of simulation results unavailable above 2000 simulations.")
    else:
        st.write("Press 'Run Simulation' to view results.")
    
if __name__ == "__main__":
    run()
