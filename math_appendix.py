import streamlit as st

def display():
  st.markdown(r""" ## Math Appendix""")

  st.markdown("""##### Retraining Costs""")
  st.markdown(r""" In the model, we assume that every time software doubles, the pace of software progress doubles $q$ times, where $q = \lambda(r^{-1} - 1)$. The underlying assumption is that models can be developed through a relatively continuous process, building on the previous model. In reality, state-of-the-art models must be trained—a process that takes time. To account for this, we assume that software improvements can also be utilized to accelerate the pace of training.
  
  To implement adjustment costs, we now require that, to double the pace of software progress, the software level must double an additional time to reduce training time. Specifically, software must double $1/q + 1$ times before the pace of software doubles. Taking the inverse (i.e., for every time software doubles, how many times does the pace of software progress double) yields $q/ q+1 $.
  
  TThis calculation assumes that software progress accelerates over time and is therefore only valid for $q > 0$. To generalize this formula to cases where software progress is slowing down ($q < 0$) we can use the expression $q/(|q| + 1)$. Why? When $q < 0$ software must double a negative number of times for the pace of software progress to double. Retraining reduces that number by 1 (since retraining still slows progress). Consequently, software must double $(1/q) - 1$ times before the pace of software progress doubles. The inverse of this is $q/(1-q)$. Therefore, we can see that taking the abolute value of $q$ in the demominator makes this adjustment robust to positive and negative values for $q$.

  In summary 
  - without retraining costs: software doubles $1/q$ times before the pace of software progress doubles; every time software doubles the pace of software progress doubles $q$ times,
  - with retraining costs: software doubles $(1/q) + 1$ times before the pace of software progress doubles; every time software doubles, the pace of software progress doubles $q/(|q|+1)$ times. """)

  st.markdown("""##### Compute Growth""")
  st.markdown(r"""When "compute growth" is selected in the model settings, it implies that the boost from deploying AI to software R&D occurs gradually over time rather than instantaneously. To implement this, we assume that the boost in each time period originates from compute growth, which grows at an exogenous rate until it reaches a ceiling. We assume this ceiling occurs after 12 doublings of compute (or a 4096× increase relative to the initial compute level). The dynamics of compute growth are given by: """)
  st.latex(r"""
  C_t = \begin{cases} C_0 \exp(g_c t) \quad \text{if } C_t < C_{\text{max}} \\ C_{\text{max}} \quad \text{otherwise}\end{cases}""")
  st.markdown(r"""where $g_c$ is the rate of growth of compute. This translates into a boost to research productivity according to:""")
  st.latex(r"""
  f(C) = f_0 + (f_{\text{max}} - f_0) \times \frac{\log(C) - \log(C_0)}{\log(C_{\text{max}}) - \log(C_0)}""")
  st.markdown(r"""Given exponential growth in compute, $f(C)$ increases linearly with time untill it reaches the compute ceiling, at which point $f$ remains at $f_{\text{max}}$. In the simulation it is assumed that $f_0 = 0.1$ so the initial bost to research productivity from deployment of AI is an additional 10% on top of the usual rate.
  """)
    
  st.markdown("""##### Connection between Doubling Time and Semi-Endogenous Growth Models""")
  st.markdown(r"""How do we derive the relationship between software doubling and software progress doubling in semi-endogenous growth models? Consider the standard semi-endogenous growth framework:""")
  st.latex(r"""
  \dot{S}_t(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{1-\lambda} r(S_t) \implies g_{S,t}(S_t, C_t) = (R(S_t, C_t))^{\lambda} S_t^{-\lambda} r(S_t)
  """)
  st.markdown(r"""Here, the first expression represents the rate of change, while the second represents the growth rate (i.e., dividing the rate of change by $S$). In contrast to the standard semi-endogenous growth framework, we replace human labor with effective researchers, $R$, and assume that the research productivity term, $r$, is a function of the software level. Since AI completely replaces human labor, effective researchers depend solely on software capabilities and compute.

  To combine software quality and compute availability into effective researchers, we assume a Cobb-Douglas form: $R(S_t, C_t) = (a S_t)^{\alpha} (b C_t)^{1-\alpha}$, where $a > 0$, $b > 0$, and $0 \leq \alpha \leq 1$.

  Next, to calculate the time required for software to double under a given growth rate, we solve for $D$ such that $2 = \exp(g_{S} \times D)$, which yields $D = \log(2) / g_{S}$. For a given doubling time, $D$, we can calculate the subsequent doubling time iteratively. For instance, holding compute fixed, let the doubling time at a given software level be $D(S)$. Then, we can show:
  """)
  st.latex(r"""
  D(2S) = D(S) \times \frac{g_{S}(S, C)}{g_{S}(2S, C)} = D(S) \times 2^{\lambda \left( \frac{1}{r(2S)} - \alpha \right)} \times S^{\lambda \left( \frac{1}{r(2S)} - \frac{1}{r(S)} \right)}
  """)
  st.markdown(r"""From this, we see that the semi-endogenous growth version of the iterative doubling time expression differs in two key ways from the one employed in the simplified model: 
  1. The semi-endogenous setup includes an $\alpha$ term, which was assumed to be 1 in the simplified model. 
  2. The semi-endogenous version includes a software level adjustment term, $S^{\lambda \left( \frac{1}{r(2S)} - \frac{1}{r(S)} \right)}$. 

  I will address each of these differences in turn.

  On point (1), this difference can be addressed by reinterpreting the $\lambda$ and $r$ terms. Instead of introducing $\alpha$ (which modifies the relationship between software levels and software progress), we reduce estimates of $\lambda$ and $r$ to account for the fact that software is only one component of "effective researchers." By defining $\hat{\lambda} = \lambda \alpha$ and $\hat{r} = r \alpha$, we see that doubling software levels increases the rate of software growth by $\hat{\lambda} \left( \frac{1}{\hat{r}(2S)} - 1 \right)$, as implemented in the simulation.

  On point (2), this difference cannot be directly rectified in the simplified model. This term arises because $r$ declines over time as the software level grows, which implies that the exponent on $S$ decreases after each doubling of software. Since $r$ is declining, we have:
  """)
  
  st.latex(r"""
  \lambda \left( \frac{1}{r(2S)} - \frac{1}{r(S)} \right) > 0
  """)
  st.markdown(r"""
  This indicates that, relative to the semi-endogenous growth model, the simplified model overestimates the doubling time according to the iterative doubling time expression above. We omit this term because it complicates the model. Instead, we treat the change in $r$ over time as a simple way to implement a penalty to additional software progress, rather than as a literal reflection of the semi-endogenous framework.
  """)

