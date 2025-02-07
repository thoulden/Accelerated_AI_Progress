import streamlit as st
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import simulation modules
import multiple_simsA 
import single_sim

# Parameters table in Markdown format
def get_parameters_table_markdown():
    table_markdown = r'''
| Parameter                                  | Description                                                         | Low Estimate  | Median Estimate | High Estimate |
|--------------------------------------------|---------------------------------------------------------------------|---------------|-----------------|---------------|
| Initial Speed Up, $f$                      | After AI Systems for AI R&D Automation (ASARA) is deployed, how many times faster does software progress become (compared to the recent pace of software progress)?  | 2             | 8               | 32            |
| $r$                       | Controls the degree of diminishing returns to research. Each time cognitive inputs to software R&D double, how many times does software double? (Note this parameter falls over time.)                | 0.4           | 1.2             | 3.6           |
| Distance to effective limits on software   | At the start of the simulation, how far is software from effective limits? Measured in the years of AI progress at recent rates of progress.         | 5             | 9               | 13            |
| Parallelizability of research, $\lambda$   | If cognitive inputs to software R&D instantaneously double, how many times does the pace of software progress double?                                    | 0.15          | 0.3             | 0.6           |
    '''
    return table_markdown

# Main Page content
st.title('Simulation of Accelerated AI Progress')

st.markdown(r"""
This tool complements the post on the pace of software progress once AI can fully replace humans in AI research.
It offers two simulation options:

- **Multiple Simulations:** Run several simulations with uncertainty over key parameters. The output is a plot showing the fraction of simulations where the growth rate of software exceeds a specified threshold over some number of years.
- **Single Simulation:** Run a single simulation under specific parameter values to illustrate the path of AI progress, including the evolution of diminishing research productivity and growth rates over time.

The simulation assumes that compute remains constant over time.
""")

st.markdown("### Results")

# Simulation Mode Selector (remains in the sidebar)
simulation_mode = st.sidebar.selectbox(
    "Select Simulation Mode",
    ("Single Simulation", "Multiple Simulations"),  # Single Simulation appears first
)

if simulation_mode == "Single Simulation":
    single_sim.run()  # Run the single simulation (placeholder)
elif simulation_mode == "Multiple Simulations":
    multiple_simsA.run()  # Run the multiple simulations (placeholder)

st.markdown("### Model Parameters and Estimates")
st.markdown(r"""
The table below summarizes the model parameters:
""")

# Display the parameters table
parameters_table_md = get_parameters_table_markdown()
st.markdown(parameters_table_md)

st.markdown(r"""
In addition to model parameters, you can select whether to enable additional model specifications:
- **Retraining Cost:** Imposes a penalty on growth by allocating some progress toward increasing the model training rate.
- **Gradual Boost:** Spreads the initial acceleration evenly over 5 years.
- **Constant Diminishing Returns:** Assumes that $r$ is fixed at its initial value over time.
""")

st.markdown("### Sampling")
st.markdown(
r"When 'Multiple Simulations' is selected, randomization occurs over log-uniform distributions for $f$, $r_0$, and $\lambda$, while the years until the ceiling are randomized over a uniform distribution. The bounds for these distributions come from the sidebar inputs."
)

