import streamlit as st

def display():
    st.markdown(r""" ## Math Appendix""")
    
    st.markdown("""##### Semi-Endogenous Growth Environment""")
    st.markdown(r""" Throughout, I assume software, $S$ follows the law of motion""")
    st.latex(r"""
    \dot{S}_t = (R_t^{ \alpha} {C}_t^{1-\alpha})^{\lambda} S_t^{1-\Beta}
    """)
    st.markdown(r""" where $R$ is the numebr of researchers (AI or human), and $C$ is the amount of compute available. Across the three cases considered (accelerate, base case, and exponential) I just vary $R_t$ and $\Beta$""")
    st.markdown("""###### Exponential Case""")
    st.markdown(r""" To achieve exponential growth in software I assume $\Beta$ remains fixed at $\beta_0$, i.e., the level of diminishing returns to research when GPT-6 is launched. Given (constant) diminishing returns to research, either the number of researchers, or the amount of compute must be growing over time. I assume it is compute growing; if we allowed (human) researchers to grow at the rate necessary to maintain exponential growth there would quickly become more human AI researchers than humans on the planet. Dividing the law of motion by software level yields the software growth rate:""")
    st.latex(r"""
    g_{S} = (R_t^{ \alpha} {C}_t^{1-\alpha})^{\lambda} S_t^{-\beta_0}
    """)
    st.markdown(r""" and to ensure that the growth rate is constant, we must have $(R_t^{ \alpha} {C}_t^{1-\alpha})^{\lambda}$ growing at the same rate $S_t^{-\Beta}$ is shrinking. I.e, """) 
    st.latex(r"""
    g_C = \frac{\beta_0}{\lambda (1-\alpha) }g_{S,0}
    """)
    st.markdown(r""" Next, we need a value for $S_0$. Since we take $g_S$ as given an this has a define expression we can back out $S_0$:""")
    st.latex(r"""
    S_0 = g_S^{\frac{-1}{\beta_0}}(\bar{R}^{ \alpha} {C}_t^{1-\alpha})^{-\frac{\lambda}{\beta_0}} 
    """)
    st.markdown(r""" To close out the model, we just assume $C_0$ is given and $R_t = \bar{R}$ for all $t$ and $\bar{R}$ is given. """)

    st.markdown("""###### Base Case""")
    st.markdown(r"""
    Now keep the same conditions as the above case (on researchers and compute growth), but allow $\Beta = \beta(S)$. I.e., the degree of diminishing returns is dependent on the software level. I assume the functional form for $\beta$ is such that every time the software level closes half the gap between its level and some software ceiling, $S_{\text{ceiling}}$, $\beta$ doubles. This functional form is given by
    """)
    st.latex(r"""
    \beta(S) = \beta_0 \left(1 - \frac{\frac{S_0}{\bar{S} }- 1}{{\frac{S_{\text{ceiling}}}{S_0} - 1}}\right)^{-1}
    """)
    st.markdown(r""" where $S_0$ is the software level at GPT-6 and $\beta_0$ is the level of diminishing returns to research at GPT-6.""") 
    st.markdown("""###### Acceleration Case""")
    st.markdown(r""" Now I allow AI to be deployed to software research, so both $R$ and $\Beta$ will vary over time ($\Beta = \beta(S)$ as in the base case). To begin, I assume software accelerates initially by a factor of $f$. To incorportate this into the growth model, I assume that the cognitive labor available from AI multiplies the number of researchers by a factor of $f^{\frac{1}{\lambda \alpha}}$. I also allow for progress in AI to offer continued increases to the stock of cognitive labor (both from humans and AI), so that $R_t = \bar{R} + \upsilon S_t$ where $\upsilon$ scales software level into effective researcher units. We now have to calibrate $\upsilon$. We can do this using the estimated rate of progress in software leading up to when GPT-6 is released, $g_{S,0}$. Namely,  """) 
    st.latex(r"""
    f^{\frac{1}{\lambda \alpha}} \bar{R} = \bar{R} + \upsilon S_0 \implies \upsilon = \bar{R} \times \left(f^{\frac{1}{\lambda \alpha}} - 1\right) \times \left[g_{S,0} \times \left(\bar{R}^\alpha C_0^{1-\alpha}\right)^{-\lambda}\right]^{\frac{1}{\beta_0}}
    """)
    st.markdown(r""" The ratio we are ultimately interested in $g_{S,\text{accel.}} / g_{S,\text{base}}$ -- an given this environment the choices of $\bar{R}$ and $C_0$ can have an effect on this ratio -- these are also the variables which seem to be difficult to calibrate meanigfully. Forutunately, changing these varibales has an (almost) undetectable effect on the ratio we are studying. 
    """)

    st.markdown("""##### Correlated Sampling""")
    st.markdown(r"""When running mutliple simulations I allow for users to select for $\beta_0$ and $f$ to be positively correlated. Specifically, I use ther relationship"
    """)
    st.latex(r"""
    \log(\beta_0) = \text{Intercept} + \text{Slope} \times \log(f) + \epsilon \quad \epsilon \sim N(0,\sigma^2)
    """)
    st.markdown(r"""
    Ignoring $\epsilon$ for a second, to calibrate the slope we just need to ensure that choices of $f$ are scaled so that each choice in the range of possible $f$ can 'pick out' a possible $\beta_0$ value. To do this, set 
    """)
    st.latex(r"""
    \text{Slope} = \frac{\log(\beta_{0,max})-\log(\beta_{0,min})}{\log(f_{max}) -\log(f_{max})}
    """)
    st.markdown(r"""Then we just have to solve for intercept that ensures (still, ignoring $\epsilon$) that picking $f_{min}$ will result in $\beta_{0,min}$ being chosen (and likewise for max values). This yields
    """)
    st.latex(r"""
    \text{Intercept} = \log(\beta_{0,min}) - \text{Slope} \times \log(f_{min})
    """)
    st.markdown(r""" In summary, under this specification, when $\epsilon = 0$ there is a one-to-one mapping from $f$ to $\beta_0$ so that a choice of $f$, situatied at some point in the distribution of possible choices of $f$ picks out a roughly comparable $\beta_0$ from the possible space of $\beta_0$. Adding $\epsilon$ then induces some variation in this process that increases with choice of $\sigma$. Note however, that if this noise results in a choice of $\beta_0$ that is outside of the bounds I allow for $\beta_0$ I just set the choice for $\beta_0$ to be the nearest acceptable value (this will result in some bunching at the edges of allowed values of $\beta_0$). I'd encourage curious users to run the 'multiple simulations' setting and display the empirical distributions to look at the observed correlation between $f$ and $\beta_0$.
    """)

    # Include a link to go back to the main page
    st.markdown("[Go back](?page=main)")
