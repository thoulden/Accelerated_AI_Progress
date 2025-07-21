import streamlit as st
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import simulation modules
import multiple_sims 
import single_sim

# Parameters table in Markdown format
st.set_page_config(
    page_title="AI Progress Sim",
    page_icon="🤖",  # You can also use an image file or URL
    layout="wide",  # Other options: "centered"
    initial_sidebar_state="expanded"  # Other options: "collapsed", "auto"
)
def get_parameters_table_markdown():
    table_markdown = r'''
| Parameter                                  | Description                                                         | Low Estimate  | Median Estimate | High Estimate |
|--------------------------------------------|---------------------------------------------------------------------|---------------|-----------------|---------------|
| Initial Speed Up ($f$)                      | After AI Systems for AI R&D Automation (ASARA) is deployed, how many times faster does software progress become (compared to the recent pace of software progress)?  | 2             | 8               | 32            |
| Returns to Software R&D ($r$)                       | Controls the degree of diminishing returns to research. Each time cognitive inputs to software R&D double, how many times does software double? (Note this parameter falls over time.)                | 0.4           | 1.2             | 3.6           |
| Distance to effective limits on software   | At the start of the simulation, how far is software from effective limits? Measured in the years of AI progress at recent rates of progress.         | 6             | 11               | 16            |
| Diminishing returns to parallel labour ($p$)   | If cognitive inputs to software R&D instantaneously double, how many times does the pace of software progress double?                           | 0.15          | 0.3             | 0.6           |
| Fraction of AI progress due to better software. | What fraction of observed progress in AI is due to software progress (rather than hardware improvements)?      | -          | 0.5             | -           |
    '''
    return table_markdown

# Main Page content
st.title('Simulation of Accelerated AI Progress')

st.markdown(r"""
This tool allows users to enter their own parameterizations for the accompanying post on the pace of AI progress.
It offers two simulation options:

- **Multiple Simulations:** Run several simulations with uncertainty over key parameters. The output is a table showing the fraction of simulations where the average pace of software progress exceeds a specified threshold over some number of years.
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
    multiple_sims.run()  # Run the multiple simulations (placeholder)

st.markdown("### Model Parameters and Estimates")
st.markdown(r"""
The table below summarizes the model parameters:
""")

# Display the parameters table
parameters_table_md = get_parameters_table_markdown()
st.markdown(parameters_table_md)

st.markdown(r"""
In addition to model parameters, you can select whether to enable additional model specifications:
- **Retraining Cost:** Reduce the degree of acceleration as some software efficiency gains are spent making training happen more quickly. 
- **Gradual Boost:** The initial speed-up from ASARA ramps up gradually over 5 years.
- **Constant Diminishing Returns:** Assumes that $r$ is fixed at its initial value over time.
""")

st.markdown("#### Sampling")
st.markdown(
r"When 'Multiple Simulations' is selected, randomization occurs over log-uniform distributions for $f$, $r_0$, and $p$, while the years until the ceiling are randomized over a uniform distribution. The bounds for these distributions come from the sidebar inputs. Note that we don't sample over the fraction of AI progress due to software progress, but instead leave this as a deterministic parameter."
)

st.markdown(r"#### Defining the inputs $r$ and $p$")
st.markdown(r"Consider the semi-endogenous software progress function:")
st.markdown(r"""$\text{Software Growth} = (L^\alpha C^{1-\alpha})^\lambda S^{-\beta}$""")
st.markdown(r"""where:
- Research investment combines human labor ($L$) and compute ($C$)
- $L$ represents human labor input
- $C$ represents computational resources
- $S$ represents the existing software stock
- α, λ, and β are parameters

From these parameters, we define our two key model inputs:
- $r$ = λα/β
- $p$ = λα

For additional details, see [LINK].""")

