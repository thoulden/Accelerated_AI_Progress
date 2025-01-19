import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

###############################################################################
#  A) CORE MODEL FUNCTIONS
###############################################################################
def sample_parameters_batch(n_samples, r_low, r_high, ly_low, ly_high, lf_low, lf_high, ib_low, ib_high, compute_growth):
    # (Same as your original sampling logic)
    initial_boost = np.exp(np.random.uniform(np.log(ib_low), np.log(ib_high), size=n_samples))
    r_initial = np.exp(np.random.uniform(np.log(r_low), np.log(r_high), size=n_samples))
    limit_years = np.random.uniform(ly_low, ly_high, size=n_samples)
    lambda_factor = np.exp(np.random.uniform(np.log(lf_low), np.log(lf_high), size=n_samples))

    factor_increase = 2  
    implied_month_growth_rate = np.log(2) / 3
    time_takes_to_factor_increase = np.log(factor_increase) / implied_month_growth_rate
    if compute_growth:
        denominator = np.full(n_samples, 1.1)
    else:
        denominator = initial_boost
    initial_factor_increase_time = time_takes_to_factor_increase / denominator

    # f_0 and f_max
    f_0 = np.full(n_samples, 0.1) if compute_growth else initial_boost
    f_max = initial_boost

    return np.column_stack((
        r_initial,                     # 0
        factor_increase * np.ones(n_samples),  # 1
        initial_factor_increase_time,  # 2
        limit_years,                   # 3
        np.full(n_samples, np.log(2)/3),  # 4 (compute_growth_monthly_rate)
        f_0,                           # 5
        f_max,                         # 6
        lambda_factor                  # 7
    ))

def dynamic_system_with_lambda(r_initial, factor_increase, initial_factor_increase_time, limit_years,
                               compute_growth_monthly_rate, f_0, f_max, lambda_factor,
                               retraining_cost, max_time_months=48):
    ceiling = 256 ** limit_years
    size = 1.0
    r = r_initial
    f = f_0

    times, sizes, rs, compute_sizes, f_values = [0], [size], [r], [1], [f]
    time_elapsed = 0
    k = r_initial / (np.log(ceiling) / np.log(factor_increase))

    current_factor_increase_time = initial_factor_increase_time

    while time_elapsed < max_time_months and size < ceiling and r > 0:
        f_old = f
        time_elapsed += current_factor_increase_time
        size *= factor_increase
        compute_size = compute_sizes[-1] * np.exp(compute_growth_monthly_rate * current_factor_increase_time)

        # Update f
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

        # Update the next factor_increase_time
        if r > 0:
            if retraining_cost:
                accel_factor = (lambda_factor*((1/r) - 1)) / (abs(lambda_factor*((1/r) - 1)) + 1)
            else:
                accel_factor = lambda_factor*(1/r - 1)
            current_factor_increase_time *= ((factor_increase**accel_factor) / ((1+f)/(1+f_old)))

    return times, sizes, rs, compute_sizes, f_values

def calculate_summary_statistics_binary(times, conditions):
    """
    Returns a dict: {(time_period, speed_up_factor): 'yes'/'no', ...}
    for the entire simulation.
    """
    results = {cond: 'no' for cond in conditions}
    for (time_period, speed_up_factor) in conditions:
        baseline_doublings = (time_period/12)*8
        required_doublings = int(baseline_doublings*speed_up_factor)
        for i in range(len(times) - required_doublings):
            time_span = times[i + required_doublings] - times[i]
            if time_span < time_period:
                results[(time_period, speed_up_factor)] = 'yes'
                break
    return results

###############################################################################
#  B) HIGH-LEVEL RUN FUNCTION
###############################################################################
def run_simulations(num_sims, conditions, r_low, r_high, ly_low, ly_high,
                    lf_low, lf_high, ib_low, ib_high,
                    retraining_cost, compute_growth):
    """
    Returns:
      - probabilities (dict of condition -> float)
      - times_matrix (list of arrays of times)
      - sizes_matrix (list of arrays of sizes)
      - stats_list (list of dicts) storing { (time_period, speed_up_factor): 'yes'/'no' }
      - params_list (list of dicts) storing parameter draws for each simulation
    """
    # 1) Sample n_samples parameter sets
    params_batch = sample_parameters_batch(
        num_sims, r_low, r_high, ly_low, ly_high,
        lf_low, lf_high, ib_low, ib_high, compute_growth
    )

    # For building final results
    times_matrix = []
    sizes_matrix = []
    stats_list = []         # each element is the "yes/no" dict for that sim
    params_list = []        # each element is the param dict for that sim

    # For tracking overall probability
    batch_summary = {cond: 0 for cond in conditions}

    progress = st.progress(0)

    for i, p in enumerate(params_batch, start=1):
        (r_init, factor_inc, init_fact_time, limit_yrs,
         comp_grow_rate, f_0, f_max, lam_factor) = p

        # 2) Run the dynamic system
        times, sizes, rs, compute_sizes, f_values = dynamic_system_with_lambda(
            r_init, factor_inc, init_fact_time, limit_yrs,
            comp_grow_rate, f_0, f_max, lam_factor,
            retraining_cost
        )

        times_matrix.append(times)
        sizes_matrix.append(sizes)

        # 3) Calculate "yes/no" stats for the entire simulation
        stats = calculate_summary_statistics_binary(times, conditions)
        stats_list.append(stats)

        # 4) Tally up how often each condition is "yes"
        for cond in conditions:
            if stats[cond] == 'yes':
                batch_summary[cond] += 1

        # 5) Save the param set for reference in CSV
        params_list.append({
            "r_initial": r_init,
            "factor_increase": factor_inc,
            "initial_factor_increase_time": init_fact_time,
            "limit_years": limit_yrs,
            "compute_growth_monthly_rate": comp_grow_rate,
            "f_0": f_0,
            "f_max": f_max,
            "lambda_factor": lam_factor
        })

        progress.progress(i/num_sims)

    # Overall probabilities
    probabilities = {
        cond: batch_summary[cond]/num_sims
        for cond in conditions
    }

    return probabilities, times_matrix, sizes_matrix, stats_list, params_list

###############################################################################
#  C) STREAMLIT MAIN APP
###############################################################################
def run():
    st.title("Run Simulations & Download Single CSV with All Data")

    run_button = st.sidebar.button("Run Simulations")

    # === 1) Sidebar inputs ===
    st.sidebar.markdown("### Key Parameter Sampling Bounds")
    ib_low = st.sidebar.number_input("Boost (f) - lower bound", min_value=0.1, value=2.0)
    ib_high = st.sidebar.number_input("Boost (f) - upper bound", min_value=ib_low, value=32.0)
    r_low = st.sidebar.number_input("Initial Productivity (r0) - lower bound", min_value=0.01, value=0.4)
    r_high = st.sidebar.number_input("Initial Productivity (r0) - upper bound", min_value=r_low, value=3.6)
    ly_low = st.sidebar.number_input("Years to Ceiling - lower bound", min_value=1.0, value=7.0)
    ly_high = st.sidebar.number_input("Years to Ceiling - upper bound", min_value=ly_low, value=14.0)
    lf_low = st.sidebar.number_input("Parallelizability (lambda) - lower bound", min_value=0.01, value=0.2)
    lf_high = st.sidebar.number_input("Parallelizability (lambda) - upper bound", min_value=lf_low, value=0.8)

    num_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=30000, value=10)
    multiples_input = st.sidebar.text_input("Growth Multiples (comma-separated)", value="3,10,30")

    retraining_cost = st.sidebar.checkbox("Retraining Cost")
    compute_growth = st.sidebar.checkbox("Compute Growth")

    # Convert user-specified multiples
    multiples = [float(m.strip()) for m in multiples_input.split(',') if m.strip()]
    # For each multiple, we check time_period in months: [1,4,12,36]
    conditions = list(product([1,4,12,36], multiples))

    if run_button:
        # 2) Run simulations
        probabilities, times_matrix, sizes_matrix, stats_list, params_list = run_simulations(
            num_sims, conditions,
            r_low, r_high, 
            ly_low, ly_high, 
            lf_low, lf_high, 
            ib_low, ib_high, 
            retraining_cost,
            compute_growth
        )

        # 3) Display summary probabilities for reference
        data_for_display = []
        for tp in sorted(set(c[0] for c in probabilities.keys())):
            row = {"Time Period (Months)": tp}
            for mult in sorted(set(c[1] for c in probabilities.keys())):
                row[f"{mult}x faster"] = probabilities.get((tp, mult), 0)
            data_for_display.append(row)
        df_prob_summary = pd.DataFrame(data_for_display)
        st.write("##### Probability that AI progress is X times faster for N months:")
        st.dataframe(df_prob_summary)

        # 4) Build ONE DataFrame that has everything:
        #    - Time-step data
        #    - Random-drawn parameters
        #    - The 'yes/no' for each condition
        #    - The sidebar input bounds (repeated)

        # Prepare column names for conditions
        # e.g. Condition " (1, 3) " => "Cond_1m_3x"
        def condition_column_name(cond):
            return f"{cond[0]}m_{cond[1]}x"  # e.g. "1m_3x"

        condition_cols = [condition_column_name(c) for c in conditions]

        # Prepare big list of rows
        all_rows = []
        for i in range(len(times_matrix)):
            sim_idx = i+1
            sim_times = times_matrix[i]
            sim_sizes = sizes_matrix[i]
            sim_stats = stats_list[i]   # dict {(time_period, multiple): 'yes'/'no'}
            sim_params = params_list[i] # dict of param draws

            # Convert sim_stats to a dict { "1m_3x": 'yes', "4m_3x": 'no', ... }
            cond_yesno_map = {}
            for cond in conditions:
                c_col = condition_column_name(cond)
                cond_yesno_map[c_col] = sim_stats[cond]

            # For each time-step, replicate param and condition info
            for t, s in zip(sim_times, sim_sizes):
                row_dict = {
                    # Simulation index
                    "Simulation": sim_idx,

                    # Time-step data
                    "Time (Months)": t,
                    "Size": s,

                    # Randomly drawn parameters for this simulation
                    "r_initial": sim_params["r_initial"],
                    "factor_increase": sim_params["factor_increase"],
                    "initial_factor_increase_time": sim_params["initial_factor_increase_time"],
                    "limit_years": sim_params["limit_years"],
                    "compute_growth_monthly_rate": sim_params["compute_growth_monthly_rate"],
                    "f_0": sim_params["f_0"],
                    "f_max": sim_params["f_max"],
                    "lambda_factor": sim_params["lambda_factor"],

                    # Sidebar bounds repeated here if you want them in each row
                    "sidebar_r_low": r_low,
                    "sidebar_r_high": r_high,
                    "sidebar_ly_low": ly_low,
                    "sidebar_ly_high": ly_high,
                    "sidebar_lf_low": lf_low,
                    "sidebar_lf_high": lf_high,
                    "sidebar_ib_low": ib_low,
                    "sidebar_ib_high": ib_high,
                    "sidebar_num_sims": num_sims,
                    "sidebar_retraining_cost": retraining_cost,
                    "sidebar_compute_growth": compute_growth
                }

                # Add the "yes/no" columns for each condition
                for c_col in condition_cols:
                    row_dict[c_col] = cond_yesno_map[c_col]

                all_rows.append(row_dict)

        # Create a single DataFrame
        df_full = pd.DataFrame(all_rows)

        # 5) Export as CSV
        csv_full = df_full.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_full,
            file_name="simulation_all_data.csv",
            mime="text/csv"
        )

    else:
        st.write("Press 'Run Simulations' to generate data.")

if __name__ == "__main__":
    run()
