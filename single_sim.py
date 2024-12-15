import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    # === Single Simulation Code ===
    # Run Simulation Button
    run_simulation = st.sidebar.button('Run Simulation')
    
    # Option to compute growth
    compute_growth = st.sidebar.checkbox('Compute Growth', value=True)

    # Parameters for the simulation
    lambda_sample = st.sidebar.number_input('Parallelizability (λ)', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    r_0_sample = st.sidebar.number_input('Initial Research Productivity (r₀)', min_value=0.0, max_value=5.0, value=0.7, step=0.1)
    Yr_Left_sample = st.sidebar.number_input('Years Till Ceiling', min_value=1.0, max_value=50.0, value=9.0, step=0.5)
    f_sample = st.sidebar.number_input('Acceleration term (f)', min_value=1.0, max_value=100.0, value=8.0, step=0.1)

    if run_simulation:
        def choose_parameters():
            """
            Choose initial parameters manually (no random sampling).
            Returns:
                r_initial: The initial value of r (diminishing returns).
                initial_doubling_time: Initial doubling time in months.
                limit_years: The limit expressed as years of progress at recent rates.
                lambda_factor: The lambda factor for adjusting doubling time.
            """
            r_initial = r_0_sample
            initial_boost = f_sample
            initial_doubling_time = 3 / initial_boost  # Assume current software doubling time is 3 months
            limit_years = Yr_Left_sample
            lambda_factor = lambda_sample
            return r_initial, initial_doubling_time, limit_years, lambda_factor

        def dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, stop_doubling_time=6, lambda_factor=0.5):
            # Convert limit_years into the actual ceiling
            ceiling = 256 ** limit_years

            r = r_initial
            doubling_time = initial_doubling_time
            size = 1.0  # Starting size
            sizes = [size]
            times = [0]  # times in months
            rs = [r]

            # Compute total_doublings
            total_doublings = int(limit_years * 8)
            k = r_initial / total_doublings  # Constant reduction in r per doubling

            time_elapsed = 0  # Track time in months

            while size < ceiling and r > 0 and doubling_time <= stop_doubling_time:
                # Update time
                time_step = doubling_time
                time_elapsed += time_step
                times.append(time_elapsed)

                # Double the size
                size *= 2
                sizes.append(size)

                # Update r
                r -= k
                rs.append(r)

                # Update the doubling time for the next iteration with lambda adjustment
                if r > 0:
                    doubling_time *= 2 ** (lambda_factor * (1 / r - 1))

            return times, sizes, rs, ceiling

        # Run the simulation
        r_initial, initial_doubling_time, limit_years, lambda_factor = choose_parameters()
        times, sizes, rs, ceiling = dynamic_system_with_lambda(r_initial, initial_doubling_time, limit_years, 6, lambda_factor)

        # Convert times to years
        times_in_years = np.array(times) / 12.0
        sizes = np.array(sizes)

        # Give ceiling a value for all time
        ceiling_time = ceiling * np.ones(len(times_in_years))
                             
        # Plot software level on a log scale (original plot)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(times_in_years, sizes, label='Software Level')
        ax1.semilogy(times_in_years, ceiling_time, 'black', linewidth=0.5)  # Ceiling line
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Software Level')
        ax1.set_title('Software Level Over Time (Log Scale)')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_yscale('log')
        ax1.legend()
        st.pyplot(fig1)

        # Calculate annualized growth rates
        growth_rates = []
        for i in range(1, len(sizes)):
            dt = times_in_years[i] - times_in_years[i-1]
            if dt > 0:
                rate = (np.log(sizes[i]) - np.log(sizes[i-1])) / dt
                growth_rates.append(rate)
            else:
                growth_rates.append(np.nan)

        growth_times = times_in_years[1:]

        # Define g and multiples
        g = 2.77
        multipliers = [3, 10, 30]

        # Plot growth rates
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(growth_times, growth_rates, label='Annualized Growth Rate', color='blue')
        ax2.axhline(y=g, color='red', linestyle='--', label=f'g = {g}')

        colors = ['green', 'orange', 'purple']
        for m, c in zip(multipliers, colors):
            ax2.axhline(y=m*g, color=c, linestyle=':', label=f'{m}x g = {m*g}')

        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Annualized Growth Rate')
        ax2.set_title('Annualized Software Growth Rate Over Time')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        st.pyplot(fig2)
