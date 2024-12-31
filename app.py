import streamlit as st
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import the other modules
import math_appendix
import multiple_simsA
import single_sim

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "main"

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Main Page"):
        st.session_state["page"] = "main"
    if st.button("Math Appendix"):
        st.session_state["page"] = "math_appendix"

# Parameters table in Markdown format
def get_parameters_table_markdown():
    table_markdown = r'''
| Parameter            | Description                                                         | Low Estimate  | Median Estimate | High Estimate |
|----------------------|---------------------------------------------------------------------|---------------|-----------------|---------------|
| $f$                  | Change in the rate of software growth after deploying AI to research  | 2             | 8               | 32            |
| $\lambda$            | Parallelizability of research                                       | 0.2           | 0.5             | 0.8           |
| r₀                   | Research productivity when GPT-6 is launched                        | 0.15          | 0.45            | 0.75          |
| Years Till Ceiling    |                                                                     | 7             | 9               | 14            |
    '''
    return table_markdown

# Main Page
if st.session_state["page"] == "main":
    st.title('Simulation of Accelerated AI Progress')

    st.markdown(r"""
    A a complement to the our post on the size and speed of a software-driven intelligence explosion, this tool is to simulate the path of AI progress once AI provides 'cognitive labor' for AI research. 
    
    This tool offers two options:
    - 'Multiple Simulations' allows you to run a bunch of simulations with uncertainty over key parameters—the output of this function will be a plot showing the fraction of simulations where the growth rate of software exceeds the observed exponential rate by some amount over some number of years. The default parameters under this simulation will give you the result reported in the table in the post.  
    - 'Single Simulation' allows you to run a single simulation under specific parameter values to illustrate the path of AI progress. Under this second option, you will also see the change in the level of diminishing research productivity over time and the growth rates.  
    
    For technical details, refer to the math appendix.
    """)

    st.markdown("### Results")

    # Simulation Mode Selector
    st.sidebar.title("Simulation Options")
    simulation_mode = st.sidebar.selectbox(
        "Select Simulation Mode",
        ("Multiple Simulations", "Single Simulation")
    )

    if simulation_mode == "Multiple Simulations":
        multiple_simsA.run()  # Placeholder for Multiple Simulations
    elif simulation_mode == "Single Simulation":
        single_sim.run()  # Placeholder for Single Simulation
    
    st.markdown("### Model Parameters and Estimates")
    st.markdown(r"""
    This table summarizes the parameters that the model relies on. Note, we assume the annual growth rate of software at GPT-6 release, $g$, is 2.77 (i.e., doubles every 3 months).
    """)

    # Display parameters table
    parameters_table_md = get_parameters_table_markdown()
    st.markdown(parameters_table_md)
    st.markdown(r"""
    In addition to model parameters you can also select whether to enable compute growth or retraining costs. Intution for these are as follows: 
    - Retraining cost: imposes a penalty on growth so that some software progress has to be 'spent' on increasing the rate at which models can be trained
    - Compute growth: this results in the initial acceleration being spread out over time so there is some initial acceleration in the rate of progress and as compute grows (up to a ceiling) there is additional boosts to research progress.
    more details on retraining costs and compute growth are provided in the math appendix. 
    """)

    st.markdown("### Sampling")
    st.markdown(
    r"When 'multiple simulations' is selected, randomization occurs over log-uniform distributions for $f$, $r_0$, $\lambda$, and Years till ceiling. "
    r"When you select 'display empirical distributions' in the sidebar, on running a simulation, you will also get a histogram of the values used in the simulation."
    )

# Math Appendix Page
elif st.session_state["page"] == "math_appendix":
    math_appendix.display()
