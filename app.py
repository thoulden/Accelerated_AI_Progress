import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the other modules
import math_appendix
import multiple_simulations
import single_simulation

# Navigation link at the top right
col1, col2 = st.columns([8, 1])
with col1:
    st.write("")  # Empty placeholder to adjust alignment
with col2:
    st.markdown("[Math Appendix](?page=math_appendix)")

## parameters to put in table
def get_parameters_table_markdown():
    table_markdown = r'''
| Parameter            | Description                                                         | Low Estimate  | Median Estimate | High Estimate |
|----------------------|---------------------------------------------------------------------|---------------|-----------------|---------------|
| $f$                  | Change in the rate of software growth after deploying AI to research  | 2             | 8               | 32            |
| $\lambda$            | Parallelizability of research                                       | 0.2           | 0.5             | 0.8           |
| $\beta_0$            | Diminishing returns in discovering new ideas when GPT-6 is launched | 0.15          | 0.45            | 0.75          |
| $\alpha$             | Contribution of cognitive labor vs. compute to AI R&D               | 0.0           | 0.5             | 1.0           |
| $g$                  | Growth rate of software when GPT-6 is deployed                      | 2.0           | 2.77            | 3.0           |
| $D$                    | Multiples of GPT-6 we can reach before a ceiling ($ D \equiv S_{\text{ceiling}} /S_0 $) | $10^{7}$ | $10^{8}$ | $10^{9}$ |
    '''
    return table_markdown



params = st.experimental_get_query_params()
if params.get('page') == ['math_appendix']:
    # Display Math Appendix
    math_appendix.display()
else:
    # Main Page Content
    st.title('Simulation of Accelerated AI Progress')

    st.markdown(r"""
    The purpose of this tool is to simulate the path of AI progress once AI provides 'cognitive labor' for AI research. Unlike standard applications of semi-endogenous growth theory, I assume that deminishing returns to (software) research effort is increasing with the level of software. 
    
    This tool offers two options:
    - 'Multiple Simulations' allows you to run a bunch of simulations with uncertainty over key parameters—the output of this function will be a plot showing the fraction of simulations where the growth rate of software exceeds the observed exponential rate by some amount over some number of years. 
    - 'Single Simulation' allows you to run a single simulation under specific parameter values to illustrate the path of AI progress. Under this second option, you will also see the change in the level of diminishing research productivity over time and the growth rates. The cases considered are the following: 
        - Exponential: This is a hypothetical case where software growth continued at the same rate as when GPT-6 is released. This is hypothetical since it ignores increasing diminishing returns to research. Exponential progress is maintained through a growing compute stock to offset (fixed) diminishing returns to research.
        - Base: Here research is left to humans, and the compute stock is growing as in the exponential case. Here, humans will still face increasing diminishing returns to research. 
        - Acelerate: Here AI is deployed to research and the compute stock is growing at the same rate as the above two cases.
    
    For technical details, refer to the math appendix.
     """)

    st.markdown("### Model Parameters and Estimates")
    st.markdown(r"""
    This table summarizes the paramaters that the model relies on. 
     """)
    # Get the parameters table in Markdown format
    parameters_table_md = get_parameters_table_markdown()

    # Display the table
    st.markdown(parameters_table_md)

    st.markdown("### Sampling")

    st.markdown(
    r"When 'multiple simulations' is selected, randomization occurs over log-uniform distributions for $f$, $\beta_0$, $\lambda$, and $D$. "
    r"When you select 'display empirical distributions' in the sidebar, on running a simulation, you will also get a histogram of the values used in the simulation."
)

    st.markdown("### Results")

    # Simulation Mode Selector
    st.sidebar.title("Simulation Options")
    simulation_mode = st.sidebar.selectbox(
        "Select Simulation Mode",
        ("Multiple Simulations", "Single Simulation")
    )

    if simulation_mode == "Multiple Simulations":
        multiple_simulations.run()
    elif simulation_mode == "Single Simulation":
        single_simulation.run()
