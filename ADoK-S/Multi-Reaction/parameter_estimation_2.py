#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:38:23 2024

@author: md1621
"""

"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
np.random.seed(1998)


"##############################################################################"
"####################### Synthetic data from case study #######################"
"##############################################################################"

# Case study
def kinetic_model(t, z):
    k_1 = 9
    k_2 = 2
    K_1a = 5
    dAdt = ((-k_1 * z[0] * z[2]) / (1 + K_1a * z[0] * z[1]))
    dBdt = ((k_1 * z[0] * z[2]) / (1 + K_1a * z[0] * z[1])) - (k_2 * z[1])
    dCdt = ((-k_1 * z[0] * z[2]) / (1 + K_1a * z[0] * z[1])) + (k_2 * z[1])
    dzdt = [dAdt, dBdt, dCdt]
    return dzdt

# Plotting the data given
species = ['A', 'B', 'C']
initial_conditions = {
    "ic_1": np.array([2.0, 1.0, 2.0]),
    "ic_2": np.array([2.0, 1.0, 0.2]),
    "ic_3": np.array([0.2, 1.0, 0.2]),
    "ic_4": np.array([2.0, 0.0, 2.0]),
    "ic_5": np.array([0.2, 0.0, 0.2]),

    "ic_6": np.array([1.10125758, 0.,         0.29795717]),
    "ic_7": np.array([2.,         0.17702532, 0.66736296]),
    "ic_8": np.array([2.,  1.,  0.2]),
    "ic_9": np.array([1.13627625, 0.,         2.        ]),
}

num_exp = len(initial_conditions)
num_species = len(species)

timesteps = 50
time = np.linspace(0, 2.5, timesteps)
t = [0, np.max(time)]
t_eval = list(time)
STD = 0.01
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
    k_5 = k[4]
    k_6 = k[5]
    k_7 = k[6]
    k_8 = k[7]
    k_9 = k[8]
    k_10 = k[9]

    # Nested function defining the system of ODEs
    def nest(t, z):
        # Differential equations for each species in the competition model
        dAdt = (-k_1 * z[0] * z[2]) / (k_2 * z[0] * z[1] + k_3 * z[0] + k_4)
        dBdt = ((k_5 * z[0] * z[2]) / (1 + k_6 * z[0] * z[1])) - (k_7 * z[1])
        # dCdt = ((-k_5 * z[0] * z[2]) / (1 + k_6 * z[0] * z[1])) + (k_7 * z[1])
        dCdt = ((-k_8 * z[0] * z[2]) / (1 + k_9 * z[0] * z[1])) + (k_10 * z[1])
        dzdt = [dAdt, dBdt, dCdt]
        return dzdt
        
    # Time points for the ODE solution
    time = np.linspace(0, 2.5, 50)
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
number_parameters = 10
lower_bound = 0.0001
upper_bound = 10

# Initial guess for the parameters
solution = np.random.uniform(lower_bound, upper_bound, number_parameters)
solution = np.array([9, 5, 0, 1, 9, 5, 2, 9, 5, 2])
print('Initial guess = ', solution)

# Perform optimization to minimize the SSE
opt_val, opt_param = Opt_Rout(multistart, number_parameters, solution, lower_bound, upper_bound, sse)

# Print the optimization results
print('MSE = ', opt_val)
print('Optimal parameters = ', opt_param)

color_1 = ['salmon', 'royalblue', 'darkviolet']

for i in range(num_exp):
    t = time
    ics = initial_conditions["ic_" + str(i + 1)]
    yy = competition(opt_param, ics)
    
    fig, ax = plt.subplots()
    # ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(t, y, "o", markersize = 4, label = species[j], color = color_1[j])
        ax.plot(t, yy[j], color = color_1[j])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha = 0.5)
    ax.legend(loc = 'upper right', fontsize = 15)

# plt.show()