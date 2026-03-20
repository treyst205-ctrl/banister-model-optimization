import numpy as np
from scipy.integrate import solve_ivp
from banister_core import banister_model, u
#To be fed into the minimize fcn
def objective_function(params,t_span, p_calculated):
    tau_f,tau_a,k1,k2,p0=params
    def fit_model(t,y): #define dupe model with open parameters
        F,D=y
        u_t=u(t)
        df=-(1/tau_f)*F+k1*u_t
        da=-(1/tau_a)*D+k2*u_t
        return [df,da]
    sol_guess=solve_ivp(fit_model,(1,200),[0,0],t_eval=t_span)
    p_pred=p0+sol_guess.y[0]-sol_guess.y[1]
    return np.sum((p_calculated-p_pred)**2) #cost fcn per guess
