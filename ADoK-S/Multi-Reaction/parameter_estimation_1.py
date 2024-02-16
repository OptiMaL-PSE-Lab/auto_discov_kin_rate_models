#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:23:02 2024

@author: md1621
"""

"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
np.random.seed(1998)


"##############################################################################"
"####################### Synthetic data from case study #######################"
"##############################################################################"

# Case study
def kinetic_model(t, z):
    k_1 = 1.4
    k_2 = 4.2
    dAdt = -k_1 * z[0]
    dBdt = k_1 * z[0] - k_2 * z[1]
    dCdt = k_2 * z[1]
    dzdt = [dAdt, dBdt, dCdt]
    return dzdt

# Plotting the data given
species = ['A', 'B', 'C']
initial_conditions = {
    "ic_1": np.array([0.5, 0.2, 0.1]),
    "ic_2": np.array([0.5, 0.2, 0.0]),
    "ic_3": np.array([0.5, 0.0, 0.1]),
    "ic_4": np.array([0.2, 0.2, 0.1]),
    "ic_5": np.array([0.2, 0.2, 0.0]),
}

num_exp = len(initial_conditions)
num_species = len(species)

timesteps = 30
time = np.linspace(0, 1, timesteps)
t = [0, np.max(time)]
t_eval = list(time)
STD = 0.0025
noise = [np.random.normal(0, STD, size = (num_species, timesteps)) for i in range(num_exp)]
in_silico_data = {}
no_noise_data = {}

for i in range(num_exp):
    ic = initial_conditions["ic_" + str(i + 1)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    in_silico_data["exp_" + str(i + 1)] = np.clip(solution.y + noise[i], 0, 1e99)
    no_noise_data["exp_" + str(i + 1)] = solution.y
    

"##############################################################################"
"######################### Optimize Kinetic Rate Model ########################"
"##############################################################################"

def competition(k, z0):
    # Define rate constants
    k_1 = k[0]
    k_2 = k[1]
    k_3 = k[2]
    k_4 = k[3]

    # Nested function defining the system of ODEs
    def nest(t, z):
        # Differential equations for each species in the competition model
        dAdt = -k_1 * z[0]
        dBdt = k_2 * z[0] - k_3 * z[1]
        dCdt = k_4 * z[1]
        dzdt = [dAdt, dBdt, dCdt]
        return dzdt
        
    # Time points for the ODE solution
    time = np.linspace(0, 1, 30)
    t = [0, np.max(time)]
    t_eval = list(time)
    
    # Solve the ODE system
    sol = solve_ivp(nest, t, z0, t_eval=t_eval, method="RK45")
    
    return sol.y

def sse(params):
    # Function to calculate Sum of Squared Errors for all experiments
    num_exp = len(initial_conditions)
    total_sse = np.zeros(num_exp)

    for i in range(num_exp):
        ic = initial_conditions["ic_" + str(i+1)]
        observations = in_silico_data["exp_" + str(i + 1)]
        model_response = competition(params, ic)

        # Calculate SSE for each experiment
        SSE = (observations - model_response)**2
        total_sse[i] = np.sum(SSE)

    return np.sum(total_sse)

def callback(xk):
    # Callback function for optimization process
    print(f"Current solution: {xk}")

def Opt_Rout(multistart, number_parameters, x0, lower_bound, upper_bound, to_opt):
    # Function to perform optimization with multiple starting points
    localsol = np.empty([multistart, number_parameters])
    localval = np.empty([multistart, 1])
    bounds = [(lower_bound, upper_bound) for _ in range(number_parameters)]
    
    for i in range(multistart):
        # Perform optimization using L-BFGS-B method
        res = minimize(to_opt, x0, method='L-BFGS-B', bounds=bounds, callback=callback)
        localsol[i] = res.x
        localval[i] = res.fun

    # Identify the best solution
    minindex = np.argmin(localval)
    opt_val = localval[minindex]
    opt_param = localsol[minindex]
    
    return opt_val, opt_param

# Setting up the optimization parameters
multistart = 10
number_parameters = 4
lower_bound = 0.0001
upper_bound = 10

# Initial guess for the parameters
solution = np.random.uniform(lower_bound, upper_bound, number_parameters)
print('Initial guess = ', solution)

# Perform optimization to minimize the SSE
opt_val, opt_param = Opt_Rout(multistart, number_parameters, solution, lower_bound, upper_bound, sse)

# Print the optimization results
print('MSE = ', opt_val)
print('Optimal parameters = ', opt_param)