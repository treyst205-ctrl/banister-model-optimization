import numpy as np

#solid, consistent, moderate rate
'''def u(t):
    return 6.0 '''
#sinusoidal oscillation from 1-10 for varying trainign intensity(use to see a how varying training stress affects the shape)
'''def u(t):
    return 5.5 + 4.5 * np.sin(2 * np.pi * t / 14)'''
#discrete training loads
'''daily_u = [40, 60, 0, 80, 50, 0, 100, 30, 0, 90, 40, 0, 110, 20, 0, 10, 5, 0, 0, 0]
days = np.arange(len(daily_u))'''

def u(t):
    #instance cycle:0-10 scale 
    #day_of_week 0=Mon 6=Sun
   if t<100:
        day_of_week = int(t) % 7 
        #intensity=intensity scale(5-100) 
        #duration=minutes(1-
    
        if day_of_week==0:    #Monday: 
            intensity=2
            duration=15
        elif day_of_week==2:  #Wednesday: 
            intensity=12.5
            duration=2
        elif day_of_week==4:  #Friday: 
            intensity=10
            duration=3
        elif day_of_week==6:  #Saturday: 
            intensity=1
            duration=7.5
        else:                   #est Days (Tue, Thu, Sun)
            intensity=0
            duration=0
        return intensity*duration #0<u(t)≤3000 per tth time
   if t>=100:
    return 0.0
   
#odes
def banister_model(t,y):
    F,D=y
    df=-(1/tau_f)*F+k1*u(t)
    da=-(1/tau_a)*D+k2*u(t)
    return [df,da]

#true solution
F_sol=sol.y[0]
D_sol=sol.y[1]


#end ODE solving