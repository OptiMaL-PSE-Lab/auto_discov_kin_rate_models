#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:00:07 2024

@author: md1621
"""

"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from sympy import *
from scipy.misc import derivative as der
import re
import itertools as it 
from time import perf_counter
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
STD = 0.01
noise = [np.random.normal(0, STD, size = (num_species, timesteps)) for i in range(num_exp)]
in_silico_data = {}
no_noise_data = {}

for i in range(num_exp):
    ic = initial_conditions["ic_" + str(i + 1)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    in_silico_data["exp_" + str(i + 1)] = np.clip(solution.y + noise[i], 0, 1e99)
    no_noise_data["exp_" + str(i + 1)] = solution.y

color_1 = ['salmon', 'royalblue', 'darkviolet']
marker = ['o', 'o', 'o']

# Plotting the in-silico data for visualisation
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(time, y, marker[j], markersize = 4, label = species[j], color = color_1[j])
    
    ax.grid(alpha = 0.5)
    ax.legend(loc='upper right', fontsize = 15)

# plt.show()


"##############################################################################"
"###################### Symbolic Regresion Concentration ######################"
"##############################################################################"

# Running symbolic regression and making csv file with best expressions for
# The concentration profiles - an expression dependent on time
# The while loop is to ensure that the algorithm runs until it gets results
# for i in range(num_exp):
#     for j in range(num_species):
#         successful = False
#         while successful is False:
#             try:
#                 if j == 0:
#                     file_name = str("Multi-Reaction/data_1/hall_of_fame_A_" + str(i + 1) + ".csv")
#                 if j == 1:
#                     file_name = str("Multi-Reaction/data_1/hall_of_fame_B_" + str(i + 1) + ".csv")
#                 if j == 2:
#                     file_name = str("Multi-Reaction/data_1/hall_of_fame_C_" + str(i + 1) + ".csv")

#                 X = time.reshape(-1, 1)
#                 Y = in_silico_data["exp_" + str(i + 1)][j].reshape(-1, 1)

#                 model = PySRRegressor(
#                     niterations = 200,
#                     binary_operators = ["+", "*", "/", "-"],
#                     unary_operators = ["exp"],
#                     model_selection = "accuracy",
#                     loss = "loss(x, y) = (x - y)^2",
#                     maxsize = 9,
#                     timeout_in_seconds = 300,
#                     parsimony = 1e-5,
#                     equation_file = file_name
#                 )

#                 model.fit(X, Y, variable_names = ["t"])
#                 successful = True
#             except:
#                 pass


"##############################################################################"
"###################### Selecting Concentration Profiles ######################"
"##############################################################################"

def read_equations(path):
    # Read equations from CSV with different separator 
    data = pd.read_csv(path)
    # Convert dataframe into numpy array
    eqs = data["Equation"].values
    
    eq_list = []
    # For every string equation in list...
    
    def make_f(eq):
        # Function takes a string equation, 
        # Converts exp to numpy representation
        # And returns the expression of that string 
        # As a function 
        def f(t):
            equation = eq.replace("x0", "t")
            return eval(equation.replace("exp", "np.exp"))
        return f
    
    for eq in eqs:
        # Iterate over expression strings and make functions
        # Then add to expression list
        eq_list += [make_f(eq)]
    
    return eq_list

def number_param(path):
    # Read equations from CSV with different separator 
    data = pd.read_csv(path)
    # Convert dataframe into numpy array
    eqs = data["Equation"].values
    t = symbols("t")
    simple_traj = []
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        simple_traj.append(func)
        things = list(func.atoms(Float))
        param.append(len(things))
    
    simple_traj = np.array(simple_traj).tolist()
    return param

def find_best_model(NLL, param):
    # Finding the model with the lowest AIC value
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def NLL_models(eq_list, t, data, NLL_species, number_datapoints):
    # Make list of NLL values for each equation
    NLL = []
    
    for f in eq_list:
        y_T = []
        
        for a in t:
            y_T.append(f(a))
        
        NLL.append(NLL_species(data, y_T, number_datapoints))
    return NLL

def NLL(C, y_C, number_datapoints):
    # Calculate the NLL value of a given equation
    likelihood = np.empty(number_datapoints)
    mse = np.empty(number_datapoints)
    
    for i in range(number_datapoints):
        mse[i] = ((C[i] - y_C[i])**2)
    
    variance = np.sum(mse) / number_datapoints
    
    for i in range(number_datapoints):
        likelihood[i] = ((C[i] - y_C[i])**2) / (2 * (variance)) \
            - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

# Find out which concentration models are best for each experiment
equation_lists = {}
best_models = {}

for i in range(num_exp):
    data = in_silico_data["exp_" + str(i + 1)]
    
    for j in range(num_species):
        if j == 0:
            file_name = str("Multi-Reaction/data_1/hall_of_fame_A_" + str(i + 1) + ".csv")
            name = "A_"
        if j == 1:
            file_name = str("Multi-Reaction/data_1/hall_of_fame_B_" + str(i + 1) + ".csv")
            name = "B_"
        if j == 2:
            file_name = str("Multi-Reaction/data_1/hall_of_fame_C_" + str(i + 1) + ".csv")
            name = "C_"
        
        a = read_equations(file_name)
        nll_a = NLL_models(a, time, data[j], NLL, timesteps)
        param_a = number_param(file_name)
        best_models[name + str(i + 1)] = find_best_model(nll_a, param_a)
        equation_lists[name + str(i + 1)] = a

# Plotting the selected concentration profile and in-silico data
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Concentration Profiles - Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        yy = equation_lists[name][model](time)
        ax.plot(time, y, marker[j], markersize = 4, label = species[j], color = color_1[j])
        ax.plot(time, yy, color = color_1[j], linestyle = "-")
    
    ax.grid(alpha = 0.5)
    ax.legend(loc='upper right', fontsize = 15)

# plt.show()


"##############################################################################"
"########################## Derivative Calculations ###########################"
"##############################################################################"

derivatives = {}
dxdt_truth = {}
SR_derivatives_A = np.array([])
SR_derivatives_B = np.array([])
SR_derivatives_C = np.array([])

# Getting the rate measurements from the model (realistically, never available)
# But just to check the fit of our estimates of the rate which are obtained by
# Numerically differentiating the concentration models selected
for i in range(num_exp):
    
    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        best_model = equation_lists[name][model]
        derivative = np.zeros(timesteps)
        
        for h in range(timesteps):
            derivative[h] =  der(best_model, time[h], dx = 1e-6)
        
        derivatives[name] = derivative
        dxdt_truth[name] = kinetic_model(time, no_noise_data["exp_" + str(i + 1)])[j]

# Plotting the estimated rates and the actual rates
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Derivative Estimates - Experiment " + str(i + 1))
    ax.set_ylabel("Rate $(Mh^{-1})$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    data = no_noise_data["exp_" + str(i + 1)]
    y = kinetic_model(time, data)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        yy = derivatives[name]
        ax.plot(time, y[j], marker[j], markersize = 4, label = species[j], color = color_1[j])
        ax.plot(time, yy, color = color_1[j], linestyle = "-")
    
    ax.grid(alpha = 0.5)
    ax.legend(loc='upper right', fontsize = 15)

# plt.show()

# Preparing the data for the second step of the symbolic regression methodology
for i in range(num_exp):
    SR_derivatives_A = np.concatenate([SR_derivatives_A, derivatives["A_" + str(i + 1)]])
    SR_derivatives_B = np.concatenate([SR_derivatives_B, derivatives["B_" + str(i + 1)]])
    SR_derivatives_C = np.concatenate([SR_derivatives_C, derivatives["C_" + str(i + 1)]])

a = in_silico_data["exp_1"].T
b = in_silico_data["exp_2"].T
SR_data = np.vstack((a, b))

for i in range(2, num_exp):
    c = in_silico_data["exp_" + str(i + 1)].T
    SR_data = np.vstack((SR_data, c))

error = {}
sr_strat = 0
for i in derivatives:
    error[i] = np.sum((derivatives[i] - dxdt_truth[i])**2)
    sr_strat += error[i]


"##############################################################################"
"########################## Symbolic Regression Rate ##########################"
"##############################################################################"

# Running symbolic regression and making csv file with the best symbolic expressions
# The kinetic rate model - an expression dependent on concentration of observed species
# The while loop is to ensure that the algorithm runs until it gets results
# for i in range(num_species):
#     successful = False
#     while successful is False:
#         try:
#             if i == 0:
#                 file_name = str("Multi-Reaction/data_1/hall_of_fame_rate_A.csv")
#                 Y = SR_derivatives_A
#                 num = 200
#             if i == 1:
#                 file_name = str("Multi-Reaction/data_1/hall_of_fame_rate_B.csv")
#                 Y = SR_derivatives_B
#                 num = 200
#             if i == 2:
#                 file_name = str("Multi-Reaction/data_1/hall_of_fame_rate_C.csv")
#                 Y = SR_derivatives_C
#                 num = 200

#             model = PySRRegressor(
#                 niterations = num,
#                 binary_operators = ["+", "*", "/", "-"],
#                 model_selection = "accuracy",
#                 loss = "loss(x, y) = (x - y)^2", 
#                 maxsize = 10,
#                 timeout_in_seconds = 300,
#                 parsimony = 1e-5,
#                 equation_file = file_name
#             )

#             model.fit(SR_data.reshape(-1, 3), Y, variable_names=["A", "B", "C"])
#             successful = True
#         except:
#             pass


"##############################################################################"
"############################ Find Best Rate Model ############################"
"##############################################################################"

# In this first part, we read the rate equations generated by symbolic regression
# And we pick the best equation for each species by evaluating them in the rate space
# Aka, we find the predicted rates and we compare with the estimated rates
# This is not the best way to do it, so essentially, ignore it
def rate_n_param(path):
    # read equations from CSV with different separator 
    data = pd.read_csv(path)
    # convert dataframe into numpy array
    eqs = data["Equation"].values
    A, B, C = symbols("A B C")
    simple_traj = []
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        func = str(func)
        j = 0
        things = re.findall(r"(\*{2}|\*{0})(\d+\.?\d*)", func)
        
        for i in range(len(things)):
            if things[i][0] != "**":
                j += 1
        
        simple_traj.append(func)
        param.append(int(j))
    # simple_traj = np.array(simple_traj).tolist()
    
    return simple_traj, param

rate_models = {}
GP_models = {}

for i in range(num_species):
    if i == 0:
        path = "Multi-Reaction/data_1/hall_of_fame_rate_A.csv"
        name_models = "A_models"
        name_params = "A_params"
    
    if i == 1:
        path = "Multi-Reaction/data_1/hall_of_fame_rate_B.csv"
        name_models = "B_models"
        name_params = "B_params"
    
    if i == 2:
        path = "Multi-Reaction/data_1/hall_of_fame_rate_C.csv"
        name_models = "C_models"
        name_params = "C_params"
    
    a, b = rate_n_param(path)
    GP_models[name_models, name_params] = a, b

def NLL_rates(rate_est, rate_pred, number_datapoints, num_exp):
    mse = (rate_est - rate_pred)**2
    variance = np.sum(mse) / (number_datapoints * num_exp)
    likelihood = ((rate_est - rate_pred)**2) / (2 * (variance)) \
        - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

def predicting_rate(equation, z):
    equation = str(equation)
    equation = equation.replace("A", "z[:, 0]")
    equation = equation.replace("B", "z[:, 1]")
    equation = equation.replace("C", "z[:, 2]")
    rate_pred = eval(equation)
    
    return rate_pred

def best_rate_model(NLL, param):
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    
    return index[0][0]

best_ODEs = {}

for i in range(num_species):
    if i == 0:
        equations, parameters = GP_models["A_models", "A_params"]
        rate_est = SR_derivatives_A
        name = "A"
    
    if i == 1:
        equations, parameters = GP_models["B_models", "B_params"]
        rate_est = SR_derivatives_B
        name = "B"
    
    if i == 2:
        equations, parameters = GP_models["C_models", "C_params"]
        rate_est = SR_derivatives_C
        name = "C"

    nll = []
    
    for equation in equations:
        rate_pred = predicting_rate(equation, SR_data)
        a = NLL_rates(rate_est, rate_pred, timesteps, num_exp)
        nll.append(a)
    
    best_ODEs[name] = best_rate_model(nll, parameters)

# In this second part, we read the symbolic expressions from the csv files, but now
# We make all possible combinations of ODEs from the proposed models and we evaluate
# Each of them and select the best one

# Here, we give make a function with a given ODE and we evaluated at a given initial condition
def rate_model(z0, equations, t, t_eval, event):
    i = 0
    
    for equation in equations:
        equation = str(equation)
        equation = equation.replace("A", "z[0]")
        equation = equation.replace("B", "z[1]")
        equation = equation.replace("C", "z[2]")
        equations[i] = equation
        i += 1
    
    def nest(t, z):
        dAdt = eval(str(equations[0]))
        dBdt = eval(str(equations[1]))
        dCdt = eval(str(equations[2]))
        dzdt = [dAdt, dBdt, dCdt]
        return dzdt
    
    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45", events = event)  
    
    return sol.y, sol.t, sol.status

equations = []
names = ["A_models", "A_params", "B_models", "B_params",\
     "C_models", "C_params"]
all_models = []
params = []

# Here we make all the possible ODEs and save the number of parameters that exists in them
for i in np.arange(0, len(names), 2):
    all_models.append(GP_models[names[i], names[i + 1]][0])
    params.append(GP_models[names[i], names[i + 1]][1])

all_ODEs = list(it.product(*all_models))
param_ODEs = list(it.product(*params))

number_models = len(all_ODEs)
AIC_values = np.zeros(number_models)

# Here we evaluate the NLL for a given ODE and experiment
def NLL_kinetics(experiments, predictions, number_species, number_datapoints):
    output = np.zeros(number_species)
    mse = np.zeros(number_species)
    variance = np.zeros(number_species)
    
    for i in range(number_species):
        a = ((experiments[i] - predictions[i])**2)
        mse[i] = np.sum(a)
        variance[i] = mse[i] / (number_datapoints)
    
    for i in range(number_species):
        likelihood = ((experiments[i] - predictions[i])**2) / (2 * (variance[i])) \
            - np.log(1 / (np.sqrt(2 * np.pi * (variance[i]))))
        output[i] = np.sum(likelihood)
    
    return np.sum(output)

# Part of solve_ivp syntax - to make sure if the ODE takes longer than 5 seconds to solve
# It gets assigned a big ol' penalty
def my_event(t, y):
    time_out = perf_counter()
    
    if (time_out - time_in) > 5:
        return 0
    
    else:
        return 1

my_event.terminal = True

# Evaluate over all possible models and experiments, save the NLL for each ODE system
# all_ODEs = GP_models["H_models", "H_params"][0]
# number_models = len(all_ODEs)
# all_ODEs = [[x] for x in all_ODEs]
AIC_values = np.zeros(number_models)

for i in range(number_models):
    neg_log = 0
    print(i)
    
    for j in range(num_exp):
        t = time
        experiments = in_silico_data["exp_" + str(j + 1)]
        time_in = perf_counter()
        ics = initial_conditions["ic_" + str(j + 1)]
        y, tt, status = rate_model(ics, list(all_ODEs[i]), [0, np.max(t)], list(t), my_event)
        
        if status == -1:
            neg_log = 1e99
            break
        
        elif status == 1:
            neg_log = 1e99
            break
        
        else:
            neg_log += NLL_kinetics(experiments, y, num_species, timesteps)
    
    num_parameters = np.sum(np.array(param_ODEs[i]))
    # num_parameters = np.sum(np.array(params[1][i]))
    AIC_values[i] = 2 * neg_log + 2 * num_parameters

# Find the best model and plot it
best_model_index = np.argmin(AIC_values)
second_min_index = np.argpartition(AIC_values, 1)[1]

for i in range(num_exp):
    t = time
    time_in = perf_counter()
    ics = initial_conditions["ic_" + str(i + 1)]
    yy, tt, _ = rate_model(ics, list(all_ODEs[best_model_index]), [0, np.max(t)], list(t), my_event)
    
    fig, ax = plt.subplots()
    # ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(t, y, "o", markersize = 4, label = species[j], color = color_1[j])
        ax.plot(tt, yy[j], color = color_1[j])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha = 0.5)
    ax.legend(loc = 'upper right', fontsize = 15)

# plt.show()

print(all_ODEs[best_model_index])
print(all_ODEs[second_min_index])
print(np.argpartition(AIC_values, 1))
