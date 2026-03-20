# Banister Fitness-Fatigue Model: Numerical Optimization
A Python-based framework for system identification and athletic performance forecasting.

## Project Structure
- `banister_core.py`: Contains the ODE system and training impulse logic.
- `optimization_tools.py`: Implements the L-BFGS-B for parameter recovery and cost function minimization.
- `main.py`: Generates synthetic noisy data, runs the optimization, and visualizes results.

## Mathematical Approach
The model represents performance ($P$) as the difference between accumulated fitness ($F$) and endured fatigue ($D$):
$$P(t) = p_0 + F(t) - D(t)$$

I utilized `scipy.optimize.minimize` to recover parameters ($\tau_f, \tau_a, k_1, k_2$) from noisy performance samples.

## How to Run
1. Run the simulation: `python main.py`# banister-model-optimization