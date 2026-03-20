import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from banister_core import banister_model, u
from optimization_tools import objective_function

# 1. Define Model Parameters
tau_f=45.0  # Fitness time constant (days)
tau_a=15.0  # Fatigue time constant (days)
k1,k2=1.0,2.0     # gain const
p0=50.0    # Baseline performance


ut_array=np.array([u(t) for t in t_span]) #training stress data
np.random.seed(42) #keep initial random value
noise=np.random.normal(loc=0,scale=1.5,size=200)
p_calculated=P_sol+noise
#print dataset
print(f"{'Day':<5} | {'Load (u)':<10} | {'P_True':<10} | {'P_Sampled':<12}")
print("-" * 45)
for i in range(len(t_span)):  # Printing first 10 days
    print(f"{t_span[i]:<5} | {ut_array[i]:<10.2f} | {P_sol[i]:<10.2f} | {p_calculated[i]:<12.2f}")
#End data gen

y0=[0.0,0.0]       
t_span=np.arange(1,201)     
t_eval=np.linspace(0,200,1000)
sol = solve_ivp(banister_model,(1,200),[0,0],t_eval=t_span)
P_sol=p0+F_sol-D_sol #performance

peak_index = np.argmax(P_sol)
peak_day = t_span[peak_index]
print("Peak performance occurs at day:", peak_day)

initial_guess=[30.0,10.0,0.5,1.5,45.0] #guess
bounds=[(20,70),(5,25),(0.1,5.0),(0.1,5.0),(40,60)] #guardrails
opt_result=minimize(objective_function,initial_guess,bounds=bounds) #leave method as default L-BFGS-B

tf_fit,ta_fit,k1_fit,k2_fit,p0_fit=opt_result.x

print("\n--- PARAMETER RECOVERY RESULTS ---")
print(f"{'Parameter':<12} | {'True Value':<12} | {'Fitted Value':<12} | {'Error':<8}")
print(f"{'-'*55}")
print(f"tau_f (Days) | {tau_f:<12.1f} | {tf_fit:<12.2f} | {abs(tau_f-tf_fit):.2f}")
print(f"tau_a (Days) | {tau_a:<12.1f} | {ta_fit:<12.2f} | {abs(tau_a-ta_fit):.2f}")
print(f"k1 (Fit Gain)| {k1:<12.1f} | {k1_fit:<12.2f} | {abs(k1-k1_fit):.2f}")
print(f"k2 (Fat Gain)| {k2:<12.1f} | {k2_fit:<12.2f} | {abs(k2-k2_fit):.2f}")
print(f"p0 (Base)    | {p0:<12.1f} | {p0_fit:<12.2f} | {abs(p0-p0_fit):.2f}")

def final_fit_model(t, y):
    return [-(1/tf_fit)*y[0]+k1_fit*u(t),-(1/ta_fit)*y[1]+k2_fit*u(t)]
sol_final=solve_ivp(final_fit_model,(1,200),[0,0],t_eval=t_span)
p_fitted_curve=p0_fit+sol_final.y[0]-sol_final.y[1]

#start example plot

plt.figure(figsize=(10, 6))
plt.plot(sol.t,F_sol,label='Fitness (F)')
plt.plot(sol.t,D_sol,label='Fatigue (A)')
plt.plot(sol.t,P_sol,label='Performance (P)')
plt.legend()
plt.title('Banister(IR) Model')
plt.xlabel('Days')
plt.ylabel('performance units')
plt.show()

#end example plot

#start sampled data plot
plt.figure(figsize=(10, 6))
plt.plot(t_span,P_sol,color='black',label='Perfect Performance(From Banister model)')
plt.scatter(t_span,p_calculated,color='orange',s=15,label='Sampled Performance(Noise)')
plt.title("Model Performance vs. Sampled Performance")
plt.xlabel("Day")
plt.ylabel("Performance Units")
plt.legend()
plt.grid(True)
plt.show()
#end sampled data plot

#start optimization plot
plt.figure(figsize=(12, 6))
plt.plot(t_span, P_sol,color='black',label='Perfect Performance(From Banister Model)')
plt.scatter(t_span,p_calculated,color='orange',s=15,label='Sampled Performance(Noise)')
plt.plot(t_span, p_fitted_curve,color='blue',label='Numerical Best Fit')
plt.title("Numerical Parameter Estimation of Banister Model")
plt.xlabel("Days")
plt.ylabel("Performance Units")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 1. Setup the 30-day window
t_forecast = np.linspace(200,230,100) 
def u_future(t):
    return 0
F_now,D_now = sol_final.y[0][-1],sol_final.y[1][-1]
def forecast_ode(t,y):
    F,D =y
    return [-(1/tf_fit)*F+k1_fit*u_future(t),-(1/ta_fit)*D+k2_fit*u_future(t)]
sol_30d=solve_ivp(forecast_ode,(200, 230),[F_now,D_now],t_eval=t_forecast)
p_30d=p0_fit+sol_30d.y[0]-sol_30d.y[1]

# Final Forecast Plotting
plt.figure(figsize=(10,6))
plt.plot(t_span, p_fitted_curve, color='blue', label='Numerical Best fit')
#Future Forecast (Day 200-230)
plt.plot(t_forecast, p_30d, color='blue', linestyle=':', linewidth=3, label='Forecasted Prediction')
plt.title("Banister Model: Crystal Ball")
plt.xlabel("Days")
plt.ylabel("Performance Units")
plt.legend()
plt.grid(True)
plt.show()