import streamlit as st

def display():
    st.markdown(r""" ## Math Appendix""")

    st.markdown("""##### Retraining Costs""")
    st.markdown(r""" In the model we assume that everytime software doubles, the pace of software progress doubles $q$ times, where $q = \lambda(r^{-1} - 1)$. The assumption underlying this process is that models can be developed through a relatively continuous process -- building on the previous model. In reality, state of the art models have to be trained, a process that takes time. To implement this possibility we assume that software improvements can also be used to increase the pace of training. 
    The median parameters suggest that software must double roughly 5 times to double the pace of software progress, to account for retraining we assume that software has to double an extra time (so roughly 6 times) to double the pace of software progress. 
    To implement this additional requirement, we assume that every time software doubles, the pace of software doubles $q/|q+1|$ times.""")

    st.markdown("""##### Connection between doubling time and semi-endogenous growth models.""")
    st.markdown(r""" How do we derive the relaitonship between software doubling and software progress doubling from semi-endogenous growth models? Take the standard semi-endogenous growth environment""")
    st.latex(r"""
    \dot{S}_t(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{1-\frac{\lambda}{r(S_t)}} \implies \dot{g}_{S,t}(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{-\frac{\lambda}{r(S_t)}}
    """)
    st.markdown(r"""In contrast to the standard semi-engodenous growth picture we replace human labor with denotes effective researchers, $R$. Since we are assuming that AI completely replaces human labor, effective researchers are just a function of software capabilities and compute.
    To combine software quality and compute availability into effective researchers we assume a Cobb-Douglass form: $R(S_t, C_t) = (a S_t)^{\alpha} (b C_t)^{1-\alpha}$. 
    Now, to answer `how much does the pace of software progress change when we double software quality (holdnding compute fixed) we just compare \dot{S}(S, C) with \dot{S}(2S, C): """)
    st.latex(r"""
    \frac{\dot{S}(2S, C)}{\dot{S}(S, C)} = \frac{[(a2S)^{\alpha}(b C)^{1-\alpha}]^{\lambda} (2S)^{-\frac{\lambda}{r}}}{[(aS)^{\alpha}(b C)^{1-\alpha}]^{\lambda} S^{-\frac{\lambda}{r(2S)}}} = 2^{\lambda \alpha}\times 2^{-\frac{\lambda}{r(2S)}}\times S^{\lambda({\frac{1}{r(S)} - \frac{1}{r(2S)}})}
    """)
    st.markdown(r""" to simplify this expression we just ignore the last $S$ term (i.e., assume here that ${\frac{1}{r(S)} =\frac{1}{r(2S)}}$), so that the rate of doubling is independent of the level of $S$. We can see that this assumption results in an overestimation of the impact of software doubling on the rate of software progress (since we are assuming that $r(2S) < r(S)$). We expect this overestimation to be relatively minor given a sufficient number of software doublings availaible before the software ceiling. 
    Under this assumetion we arive at the result that a doubling of software doubles the rate of software progress $\lambda(\alpha - \frac{1}{r(2S)})$ times, or equivalently, by $\lambda\alpha(1 - \frac{1}{r(2S)\alpha})$ times. In the original post we (generally) assume that compute is held fixed. So instead of introducing $\alpha$ (which does impact the effect of software levels on sofftware progress) we opt to just reduce estimates of $\lambda$ and $r$ to account for the fact that software is only one component of `effective researchers'. Defining $\hat{\lambda} = \lambda \alpha$ and $\hat{r} = r \alpha$ we can see that we arrive at the result that doubling software levels doubdle the rate of software growth by $\hat{\lambda}(1 - \frac{1}{\hat{r}(2S)})$. 
    """)
    
