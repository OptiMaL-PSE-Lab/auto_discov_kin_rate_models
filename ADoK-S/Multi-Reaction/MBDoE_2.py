#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:37:22 2024

@author: md1621
"""

"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from scipy.optimize import minimize

np.random.seed(1998)


"##############################################################################"
"###################### Model-Based Design of Experiments #####################"
"##############################################################################"

def SR_model(z0, equations, t, t_eval):
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
    
    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45")  
    
    return sol.y

def MBDoE(ic, time, sym_model_1, sym_model_2):
    timesteps = len(time)
    SR_thing_1 = SR_model(ic, sym_model_1, [0, np.max(time)], list(time))
    SR_thing_1 = SR_thing_1.reshape(len(time), -1)
    SR_thing_2 = SR_model(ic, sym_model_2, [0, np.max(time)], list(time))
    SR_thing_2 = SR_thing_2.reshape(len(time), -1)
    difference = -np.sum((SR_thing_1 - SR_thing_2)**2)
    return difference

def Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    time, sym_model_1, sym_model_2):
    localsol = np.empty([multistart, number_parameters])
    localval = np.empty([multistart, 1])
    boundss = tuple([(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))])
    
    for i in range(multistart):
        x0 = np.random.uniform(lower_bound, upper_bound, size = number_parameters)
        res = minimize(to_opt, x0, args = (time, sym_model_1, sym_model_2), \
                        method = 'L-BFGS-B', bounds = boundss)
        localsol[i] = res.x
        localval[i] = res.fun

    minindex = np.argmin(localval)
    opt_val = localval[minindex]
    opt_param = localsol[minindex]
    
    return opt_val, opt_param


"##############################################################################"
"########################## MBDoE on Competing Models #########################"
"##############################################################################"

multistart = 1
number_parameters = 3
lower_bound = np.array([0.2, 0.0, 0.2])
upper_bound = np.array([2.0, 1.0, 2.0])
to_opt = MBDoE
timesteps = 50
time = np.linspace(0, 2.5, timesteps)

sym_model_1 = list((
    '-2.5449934*A*C/(A*(B + 0.22843383) + 0.33003107)', '1.6521204*(A*C - B*(A*B + 0.4088003))/(A*B + 0.4088003)', '1.6029888*(-A*C + B*(A*B + 0.4037924))/(A*B + 0.4037924)'
))

sym_model_2 = list((
    '-2.5449934*A*C/(A*(B + 0.22843383) + 0.33003107)', '(2.00475475791908*A*C - 1.8638716*B*(A*(B + 0.16816978) + 0.22809806))/(A*(B + 0.16816978) + 0.22809806)', '1.6029888*(-A*C + B*(A*B + 0.4037924))/(A*B + 0.4037924)'
)) 

# THESE MODELS AND THE REAL ONE ARE INDISTINGUISHABLE WITHIN BOUNDS
a, b = Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    time, sym_model_1, sym_model_2)

print('Optimal experiment: ', b)


"##############################################################################"
"########################### Plot MBDoE Experiment ############################"
"##############################################################################"

Title = "MBDoE Second Best SR Model vs Real Model"
species = ["A", "B", "C"]

STD = 0.
noise = np.random.normal(0, STD, size = (number_parameters, timesteps))

y   = SR_model(b, sym_model_1 , [0, np.max(time)], list(time))
yyy = SR_model(b, sym_model_2, [0, np.max(time)], list(time))

fig, ax = plt.subplots()
ax.set_title(Title)
ax.set_ylabel("Concentration $(M)$")
ax.set_xlabel("Time $(h)$")

color_1 = cm.viridis(np.linspace(0, 1, number_parameters))
color_2 = cm.Wistia(np.linspace(0, 1, number_parameters))
color_3 = cm.cool(np.linspace(0, 1, number_parameters))

for j in range(number_parameters):
    # ax.plot(time, np.clip(y[j] + noise[j], 0, 1e99), "x", markersize = 3, color = color_1[j])
    ax.plot(time, y[j], color = color_1[j], label = str('SR Model 1 - ' + str(species[j])))
    ax.plot(time, yyy[j], linestyle = 'dashed', color = color_1[j], label = str('SR Model 2 - ' + str(species[j])))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(alpha = 0.5)
ax.legend()
plt.show()

print(np.sum((y - yyy)**2))