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
| $f$                  | After ASARA is deployed, how many times faster does software progress become?  | 2             | 8               | 32            |
| $\lambda$            | Parallelizability of research                                       | 0.15           | 0.3             | 0.6           |
| r₀                   | (Initially) Each time cognitive inputs to software R&D double, how many times does software double?                 | 0.4          | 1.2           | 3.6          |
| Distance to physical software limits    |     At the start of the simulation, how far is software from effective limits? Measured in the years of AI progress at recent rates of progress.         | 5             | 9               | 13            |
    '''
    return table_markdown

# Main Page
if st.session_state["page"] == "main":
    st.title('Simulation of Accelerated AI Progress')

    st.markdown(r"""
    This tool is a complement to the this post on the pace of software porgress once AI can totally replace humans in AI reseasrch.
    This tool offers two options:
    - 'Multiple Simulations' allows you to run a bunch of simulations with uncertainty over key parameters—the output of this function will be a plot showing the fraction of simulations where the growth rate of software exceeds the observed exponential rate by some amount over some number of years. The default parameters under this simulation will give you the result reported in the table in the post.  
    - 'Single Simulation' allows you to run a single simulation under specific parameter values to illustrate the path of AI progress. Under this second option, you will also see the change in the level of diminishing research productivity over time and the growth rates.  
    
    For technical details, refer to the math appendix.
    """)

    st.markdown("### Results")

    # Simulation Mode Selector
    simulation_mode = st.sidebar.selectbox(
    "Select Simulation Mode",
    ("Single Simulation", "Multiple Simulations"),  # Single Sim first
    )

    if simulation_mode == "Single Simulation":
        single_sim.run()  # Placeholder for Single Simulations
    elif simulation_mode == "Multiple Simulations":
        multiple_simsA.run()  # Placeholder for Multiple Simulation
    
    st.markdown("### Model Parameters and Estimates")
    st.markdown(r"""
    This table summarizes the parameters of the model:
    """)

    # Display parameters table
    parameters_table_md = get_parameters_table_markdown()
    st.markdown(parameters_table_md)
    st.markdown(r"""
    In addition to model parameters you can also select whether to enable compute growth or retraining costs. Details on the specifics of these choices can be found in the math appendix, but briefly: 
    - Retraining cost imposes a penalty on growth so that some software progress has to be 'spent' on increasing the rate at which models can be trained
    - Compute growth results in the initial acceleration being spread out over time so there is some initial acceleration in the rate of progress and as compute grows (up to a ceiling) there is additional boosts to research progress.
   
    """)

    st.markdown("### Sampling")
    st.markdown(
    r"When 'multiple simulations' is selected, randomization occurs over log-uniform distributions for $f$, $r_0$, and $\lambda$; while years till ceiling is randomized over a uniform distribution. The (maximum/minimum) bounds on this distribution come from the input on the sidebar. "
    )

# Math Appendix Page
elif st.session_state["page"] == "math_appendix":
    math_appendix.display()
