import streamlit as st

def display():
  st.markdown(r""" ## Math Appendix""")

  st.markdown("""##### Retraining Costs""")
  st.markdown(r""" In the model, we assume that every time software doubles, the pace of software progress doubles $q$ times, where $q = \lambda(r^{-1} - 1)$. The assumption underlying this process is that models can be developed through a relatively continuous process, building on the previous model. In reality, state-of-the-art models must be trained, a process that takes time. To implement this possibility, we assume that software improvements can also be used to increase the pace of training.  

  To implement adjustment costs we now require that to double the pace of software progress the software level must double an extra time to shorten training time, i.e., software must double $1/q + 1$ times before the pace of software doubles. Or taking the inverse (for every time that software doubles, how many times does the pace of software progress double?) yeilds $q/ q+1 $
  
  This calculation assumes that software progress is getting faster over time so it is only valid for $q > 0$. To generalize this formula to cases where software progress is slowing over time ($q < 0$) we can use the expression $q/(|q| + 1)$. Why? When $q < 0$ software must double a negative number of times for the pace of software progress to double, retraining reduces that number by 1 (because retraining still makes progress change more slowly). So now software must double $(1/q) - 1$ times before the pace of software progress doubles. The inverse of this is $q/(1-q)$. Therefore, we can see that taking the abolute value of $q$ in the demominator makes this adjustment robust to positive and negative values for $q$.

  In summary 
  - without retraining costs: software doubles $1/q$ times before the pace of software progress doubles; every time software doubles the pace of software progress doubles $q$ times,
  - with retraining costs: software doubles $(1/q) + 1$ times before the pace of software progress doubles; every time software doubles, the pace of software progress doubles $q/(|q|+1)$ times. """)

  st.markdown("""##### Compute Growth""")
  st.markdown(r"""When 'compute growth' is selected in the model settings this means that the boost from deploying AI to software R&D happens over time, rather than all in one go. To implement this we assume that the boost in each time period originates from compute growth, which is growing an an exogenous rate until it hits a ceiling; we assume this ceiling is after 12 doublings of compute (or 4096x the original size of compute). """)
  st.latex(r"""
  C_t = \begin{cases} C_0 \exp(g_c t) \quad \text{if } C_t < C_{\text{max}} \\ C_{\text{max}} \quad \text{otherwise}\end{cases}""")
  st.markdown(r"""where $g_c$ is the rate of growth of compute. This translates into a boost to research productivity according to:""")
  st.latex(r"""
  f(C) = f_0 + (f_{\text{max}} - f_0) \times \frac{\log(C) - \log(C_0)}{\log(C_{\text{max}}) - \log(C_0)}""")
  st.markdown(r"""Given exponential growth in compute, $f(C)$ increases linearly with time untill it reaches the compute ceiling, at which point $f$ remains at $f_{\text{max}}$. In the simulation it is assumed that $f_0 = 0.1$ so the initial bost to research productivity from deployment of AI is an additional 10% on top of the usual rate.
  """)
    
  st.markdown("""##### Connection between Doubling Time and Semi-Endogenous Growth Models""")
  st.markdown(r""" How do we derive the relationship between software doubling and software progress doubling from semi-endogenous growth models? Take the standard semi-endogenous growth environment:""")
  st.latex(r"""
  \dot{S}_t(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{1-\frac{\lambda}{r(S_t)}} \implies {g}_{S,t}(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{-\frac{\lambda}{r(S_t)}}
  """)
  st.markdown(r"""Where the first expression is the rate of and the second expression is the growth rate (dividing the rate of change expression by $S$). In contrast to the standard semi-endogenous growth framework, we replace human labor with effective researchers, $R$, and assume that the research productivity term, $r$, is a function of software level. Since we assume that AI completely replaces human labor, effective researchers are just a function of software capabilities and compute.

  To combine software quality and compute availability into effective researchers, we assume a Cobb-Douglas form: $R(S_t, C_t) = (a S_t)^{\alpha} (b C_t)^{1-\alpha}$, for some $a$ and $b > 0$ and $0 \leq \alpha \leq 1$. 

  Next, to calculate the time it takes for software to double under a given growth rate we solve for the $D$ such that $2 = \exp(g_{S}\times D)$, which yields $D = \log(2)/g_{S}$. Next, for a given doubling time, $D$ we can calculate the time it takes to do the subsequent doubling so that we can write out the doubling time itteratively. For example, holding compute fixed, call doubling time $D(S)$ at a given software level we can show 
  """)
  st.latex(r"""D(2S) = D(S) \times \frac{g_{S}(S,C)}{g_{S}(2S,C)} = D(S) \times 2^{\lambda( \frac{1}{r(2S)} - \alpha)} \times S^{\lambda( \frac{1}{r(2S)} -  \frac{1}{r(S)})}""")
  st.markdown(r""" Hence we can see that the semi-endogenous growth version of the itterative doubling time expression differs in two ways from the one employed in the post: i). the semi-engodenous set up includes an $\alpha$ term which was equal to one in the post and ii). the semi-endogenous version include a software level adjustment term, $S^{\lambda(\frac{1}{r(2S)} - \frac{1}{r(S)})}$. I will say something about each of these differences in turn.

  On i), this difference can be recitfied by reinterpretting $\lambda$ and $r$ terms. Instead of introducing $\alpha$ (which does affect the relationship between software levels and software progress), we opt to reduce estimates of $\lambda$ and $r$ to account for the fact that software is only one component of "effective researchers." Defining $\hat{\lambda} = \lambda \alpha$ and $\hat{r} = r \alpha$, we can see that doubling software levels doubles the rate of software growth by $\hat{\lambda}(\frac{1}{\hat{r}(2S)}-1)$, as implemented in the simulation.

  On ii), this difference can't be rectified directly with our simple model. We can see that this term comes from the fact that $r$ is declining over time as software level grows and which implies that the exponent on $S$ after a doubling of software will decrease. Because $r$ is declining, we have that $\lambda(\frac{1}{r(2S)} - \frac{1}{r(S)}) > 0$ which implies that, relative to the semi-engodenous growth model, our simple model will over estimate the doubling time according to the itterative doubling time expression above. We omit this terms since it complicates the model. We prefer for the changing in $r$ over time to be thought of as a simple way of implementing a penalty to additional software progress, rather than raken too litterally as we in the semi-endogenous version of this model detailed above.   
  """)
