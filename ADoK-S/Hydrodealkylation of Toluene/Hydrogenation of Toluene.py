#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:26:13 2022

@author: md1621
"""


"################### Importing important packages ###################"

from symbol import parameters
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy import interpolate
from pysr import PySRRegressor
from scipy.misc import derivative
from matplotlib.pyplot import cm
import pandas as pd 
from sympy import *
import re
np.random.seed(1998)


"################### Synthetic data from case study ###################"

# function that returns dz/dt
def kinetic_model(z,t):
    k_1 = 2 
    k_2 = 9 
    k_3 = 5
    dTdt = (-1)*((k_1*z[1]*z[0])/(1+k_2*z[2]+k_3*z[0]))
    dHdt = (-1)*((k_1*z[1]*z[0])/(1+k_2*z[2]+k_3*z[0]))
    dBdt = ((k_1*z[1]*z[0])/(1+k_2*z[2]+k_3*z[0]))
    dMdt = ((k_1*z[1]*z[0])/(1+k_2*z[2]+k_3*z[0]))
    dzdt = [dTdt,dHdt,dBdt,dMdt]
    return dzdt

# time points
number_exp = 3
number_datapoints = 50
t = np.linspace(0,10,number_datapoints)
CT_beg = 1
CT_end = 5
CH_beg  = 3
CH_end = 8
CB_beg = 0
CB_end = 2
CM_beg = 0.5
CM_end = 3

initial_conditions_T = np.linspace(CT_beg,CT_end,number_exp)
initial_conditions_H = np.linspace(CH_beg,CH_end,number_exp)
initial_conditions_B = np.linspace(CB_beg,CB_end,number_exp)
initial_conditions_M = np.linspace(CM_beg,CM_end,number_exp)
initial_conditions = np.vstack([initial_conditions_T,initial_conditions_H,\
                                initial_conditions_B,initial_conditions_M])
initial_conditions = np.c_[initial_conditions,np.array([5,7.3022566,0,2.24662631])]
# initial_conditions = np.c_[initial_conditions,np.array([5,3.33560455,0.8432168,0.73896702])]
number_exp = 4
# for i in range(initial_conditions.shape[0]):
#     np.random.shuffle(initial_conditions[i])

# initial_conditions = np.array([[10,10,12],[19,19,19],[6,2,4]])

# Get the rate
k_1 = 2 
k_2 = 9 
k_3 = 5
num_species = np.shape(initial_conditions)[0]
# rate = np.empty([number_exp,len(t)])
z = np.empty([number_exp,len(t),num_species])
noisy_data = np.empty([number_exp,len(t),num_species])
noise_T = np.empty([number_exp,len(t)])
noise_H = np.empty([number_exp,len(t)])
noise_B = np.empty([number_exp,len(t)])
noise_M = np.empty([number_exp,len(t)])
STD_T = 0.05339265
STD_H = 0.17873967
STD_B = 0.14714748
STD_M = 0.19606988

# different experiments at different initial conditions for A (maybe change B also?)
for i in range(number_exp):
    # initial condition
    # step_size = (CA_end - CA_beg) / (number_exp - 1)
    # res = CA_beg / step_size
    z0 = initial_conditions[:,i]
    # p = int(j/step_size - res + 1e-5)
    
    noise_T[i] = np.random.normal(0,STD_T,len(t))
    noise_H[i] = np.random.normal(0,STD_H,len(t))
    noise_B[i] = np.random.normal(0,STD_B,len(t))
    noise_M[i] = np.random.normal(0,STD_M,len(t))
    # noise_NO[i] = np.zeros(len(t))
    # noise_N[i] = np.zeros(len(t))
    # noise_O[i] = np.zeros(len(t))
    
    # solve ODE
    z[i] = odeint(kinetic_model,z0,t)
    noisy_data[i][:,0] = z[i][:,0] + noise_T[i]
    noisy_data[i][:,1] = z[i][:,1] + noise_H[i]
    noisy_data[i][:,2] = z[i][:,2] + noise_B[i]
    noisy_data[i][:,3] = z[i][:,3] + noise_M[i]
    
for a in range(number_exp):
    for b in range(len(t)):
        for c in range(num_species):
            if noisy_data[a][b][c] < 0:
                noisy_data[a][b][c] = 0

grand_exp_data = np.reshape(noisy_data,((len(t)*number_exp),num_species))
exp_l = ["1","2","3","4","5","6"]   

for i in range(initial_conditions.shape[1]):
    fig, ax = plt.subplots()
    ax.set_title('Experiment '+exp_l[i])
    ax.set_ylabel('Concentrations $(M)$')
    ax.set_xlabel('Time $(s)$')

    ax.plot(t,z[i][:,0],color='royalblue',label='Toluene')
    ax.plot(t,z[i][:,1],color='lightskyblue',label='Hydrogen')
    ax.plot(t,z[i][:,2],color='lightsalmon',label='Benzene')
    ax.plot(t,z[i][:,3],color='tomato',label='Methane')
    ax.plot(t,noisy_data[i][:,0],'o',markersize=2,color='royalblue')
    ax.plot(t,noisy_data[i][:,1],'o',markersize=2,color='lightskyblue')
    ax.plot(t,noisy_data[i][:,2],'o',markersize=2,color='lightsalmon')
    ax.plot(t,noisy_data[i][:,3],'o',markersize=2,color='tomato')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.set_xticks(np.arange(0,51,10))
    # ax.set_yticks(np.arange(0,21,4))
    # ax.axis(xmin=0,xmax=51)
    # ax.axis(ymin=0,ymax=21)
    ax.grid(alpha=0.5)
    ax.legend()
    name_fig = str('Experiment_'+exp_l[i]+'_HydroTol.png')
    plt.savefig(name_fig,dpi=600)
plt.show()


"##################### Symbolic Regresion Concentration #####################"

# profiles = ['CT1','CH1','CB1','CM1','CT2','CH2','CB2','CM2',\
#             'CT3','CH3','CB3','CM3','CT4','CH4','CB4','CM4']
# entries = ['00','01','02','03','10','11','12','13',\
#            '20','21','22','23','30','31','32','33']    

# for i in range(len(profiles)):
#     successful = False
#     while successful is False:
#         try:
#             file_name = str('hall_of_fame_' + profiles[i] + '.csv')
#             experiment = int(entries[i][0])
#             species = int(entries[i][1])
            
#             model = PySRRegressor(
#                 niterations=200,
#                 binary_operators=["+", "*", "/", "-"],
#                 unary_operators=["exp"],
#                 model_selection="accuracy",
#                 loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
#                 maxsize = 25,
#                 timeout_in_seconds = 300,
#                 parsimony = 1e-5,
#                 equation_file= file_name,
#                 variable_names = ["t"],
#             )
            
# #             model.fit(t.reshape(-1,1),noisy_data[experiment][:,species])
#             model.fit(t.reshape(-1,1),noisy_data[experiment][:,species],variable_names=["t"]) 
#             successful = True
#         except:
#             pass


"##################### Selecting Concentration Profiles 1 #####################"

def read_equations(path):
    # read equations from CSV with different separator 
    # data = pd.read_csv(path,sep='|')
    data = pd.read_csv(path)
    # convert dataframe into numpy array
    eqs = data['Equation'].values
    
    eq_list = []
    #for every string equation in list...
        
    def make_f(eq):
        # function takes a string equation, 
        # converts exp to numpy representation
        # and returns the expression of that string 
        # as a function 
        def f(t):
            equation = eq.replace('x0','t')
            return eval(equation.replace('exp','np.exp'))
        return f
    
    for eq in eqs:
        # iterate over expression strings and make functions
        # then add to expression list
        eq_list += [make_f(eq)]
        
    return eq_list

def number_param(path):
    # read equations from CSV with different separator 
    # data = pd.read_csv(path,sep='|')
    data = pd.read_csv(path)
    # convert dataframe into numpy array
    eqs = data['Equation'].values
    t = symbols('t')
    simple_traj = []
    param = []
    for eq in eqs:
        func = simplify(eq)
        simple_traj.append(func)
        things = list(func.atoms(Float))
        param.append(len(things))
    simple_traj = np.array(simple_traj).tolist()
    return param

def find_best_model(NLL,param):
    AIC = 2*np.array(NLL) + 2*np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def plot_eq_list(xlims,eq_list):
    plt.figure()
    x_plot = np.linspace(xlims[0],xlims[1],100)
    for f in eq_list:
        y_plot = []
        for xp in x_plot:
            y_plot.append(f(xp))
        plt.plot(x_plot,y_plot)
    plt.show()
    return 

def NLL_T(T,y_T,variance):
    likelihood = np.empty(number_datapoints)
    for i in range(number_datapoints):
        likelihood[i] = ((T[i] - y_T[i])**2)/(2*(variance[0]**2)) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[0]**2))))
    return np.sum(likelihood)

def NLL_models(eq_list,t,data,variance,NLL_species):
    NLL = []
    for f in eq_list:
        y_T = []
        for a in t:
            y_T.append(f(a))
        NLL.append(NLL_species(data,y_T,variance))
    return NLL

eq_list_CT1 = read_equations('hall_of_fame_CT1.csv')
eq_list_CH1 = read_equations('hall_of_fame_CH1.csv')
eq_list_CB1 = read_equations('hall_of_fame_CB1.csv')
eq_list_CM1 = read_equations('hall_of_fame_CM1.csv')

t = np.linspace(0,10,number_datapoints)
variance = np.array([STD_T*2,STD_H*2,STD_B*2,STD_M*2])
# variance = np.array([0.0001,0.0001,0.0001])


y_1 = noisy_data[0] 


T_1 = y_1[:,0]
print("Toluene Experiment 1")
NLL_T1 = NLL_models(eq_list_CT1,t,T_1,variance,NLL_T)
param_T1 = number_param('hall_of_fame_CT1.csv')
AIC_T1 = find_best_model(NLL_T1,param_T1)
print(AIC_T1)


def NLL_H(H,y_H,variance):
    likelihood = np.empty(number_datapoints)
    for i in range(number_datapoints):
        likelihood[i] = ((H[i] - y_H[i])**2)/(2*(variance[1]**2)) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[1]**2))))
    return np.sum(likelihood)


H_1 = y_1[:,1]
print("Hydrogen Experiment 1")
NLL_H1 = NLL_models(eq_list_CH1,t,H_1,variance,NLL_H)
param_H1 = number_param('hall_of_fame_CH1.csv')
AIC_H1 = find_best_model(NLL_H1,param_H1)
print(AIC_H1)


def NLL_B(B,y_B,variance):
    likelihood = np.empty(number_datapoints)
    for i in range(number_datapoints):
        likelihood[i] = ((B[i] - y_B[i])**2)/(2*(variance[2]**2)) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[2]**2))))
    return np.sum(likelihood)


B_1 = y_1[:,2]
print("Benzene Experiment 1")
NLL_B1 = NLL_models(eq_list_CB1,t,B_1,variance,NLL_B)
param_B1 = number_param('hall_of_fame_CB1.csv')
AIC_B1 = find_best_model(NLL_B1,param_B1)
print(AIC_B1)

def NLL_M(M,y_M,variance):
    likelihood = np.empty(number_datapoints)
    for i in range(number_datapoints):
        likelihood[i] = ((M[i] - y_M[i])**2)/(2*(variance[2]**2)) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[2]**2))))
    return np.sum(likelihood)


M_1 = y_1[:,3]
print("Methane Experiment 1")
NLL_M1 = NLL_models(eq_list_CM1,t,M_1,variance,NLL_M)
param_M1 = number_param('hall_of_fame_CM1.csv')
AIC_M1 = find_best_model(NLL_M1,param_M1)
print(AIC_M1)


"##################### Selecting Concentration Profiles 2 #####################"

t = np.linspace(0,10,number_datapoints)

y_2 = noisy_data[1]

eq_list_CT2 = read_equations('hall_of_fame_CT2.csv')
eq_list_CH2 = read_equations('hall_of_fame_CH2.csv')
eq_list_CB2 = read_equations('hall_of_fame_CB2.csv')
eq_list_CM2 = read_equations('hall_of_fame_CM2.csv')

T_2 = y_2[:,0]
print("Toluene Experiment 2")
NLL_T2 = NLL_models(eq_list_CT2,t,T_2,variance,NLL_T)
param_T2 = number_param('hall_of_fame_CT2.csv')
AIC_T2 = find_best_model(NLL_T2,param_T2)
print(AIC_T2)

H_2 = y_2[:,1]
print("Hydrogen Experiment 2")
NLL_H2 = NLL_models(eq_list_CH2,t,H_2,variance,NLL_H)
param_H2 = number_param('hall_of_fame_CH2.csv')
AIC_H2 = find_best_model(NLL_H2,param_H2)
print(AIC_H2)

B_2 = y_2[:,2]
print("Benzene Experiment 2")
NLL_B2 = NLL_models(eq_list_CB2,t,B_2,variance,NLL_B)
param_B2 = number_param('hall_of_fame_CB2.csv')
AIC_B2 = find_best_model(NLL_B2,param_B2)
print(AIC_B2)

M_2 = y_2[:,3]
print("Methane Experiment 2")
NLL_M2 = NLL_models(eq_list_CM2,t,M_2,variance,NLL_M)
param_M2 = number_param('hall_of_fame_CM2.csv')
AIC_M2 = find_best_model(NLL_M2,param_M2)
print(AIC_M2)


"##################### Selecting Concentration Profiles 3 #####################"

t = np.linspace(0,10,number_datapoints)

y_3 = noisy_data[2]

eq_list_CT3 = read_equations('hall_of_fame_CT3.csv')
eq_list_CH3 = read_equations('hall_of_fame_CH3.csv')
eq_list_CB3 = read_equations('hall_of_fame_CB3.csv')
eq_list_CM3 = read_equations('hall_of_fame_CM3.csv')

T_3 = y_3[:,0]
print("Toluene Experiment 3")
NLL_T3 = NLL_models(eq_list_CT3,t,T_3,variance,NLL_T)
param_T3 = number_param('hall_of_fame_CT3.csv')
AIC_T3 = find_best_model(NLL_T3,param_T3)
print(AIC_T3)

H_3 = y_3[:,1]
print("Hydrogen Experiment 3")
NLL_H3 = NLL_models(eq_list_CH3,t,H_3,variance,NLL_H)
param_H3 = number_param('hall_of_fame_CH3.csv')
AIC_H3 = find_best_model(NLL_H3,param_H3)
print(AIC_H3)

B_3 = y_3[:,2]
print("Benzene Experiment 3")
NLL_B3 = NLL_models(eq_list_CB3,t,B_3,variance,NLL_B)
param_B3 = number_param('hall_of_fame_CB3.csv')
AIC_B3 = find_best_model(NLL_B3,param_B3)
print(AIC_B3)

M_3 = y_3[:,3]
print("Methane Experiment 3")
NLL_M3 = NLL_models(eq_list_CM3,t,M_3,variance,NLL_M)
param_M3 = number_param('hall_of_fame_CM3.csv')
AIC_M3 = find_best_model(NLL_M3,param_M3)
print(AIC_M3)


"##################### Selecting Concentration Profiles 4 #####################"

t = np.linspace(0,10,number_datapoints)

y_4 = noisy_data[3]

eq_list_CT4 = read_equations('hall_of_fame_CT4.csv')
eq_list_CH4 = read_equations('hall_of_fame_CH4.csv')
eq_list_CB4 = read_equations('hall_of_fame_CB4.csv')
eq_list_CM4 = read_equations('hall_of_fame_CM4.csv')

T_4 = y_4[:,0]
print("Toluene Experiment 4")
NLL_T4 = NLL_models(eq_list_CT4,t,T_4,variance,NLL_T)
param_T4 = number_param('hall_of_fame_CT4.csv')
AIC_T4 = find_best_model(NLL_T4,param_T4)
print(AIC_T4)

H_4 = y_4[:,1]
print("Hydrogen Experiment 4")
NLL_H4 = NLL_models(eq_list_CH4,t,H_4,variance,NLL_H)
param_H4 = number_param('hall_of_fame_CH4.csv')
AIC_H4 = find_best_model(NLL_H4,param_H4)
print(AIC_H4)

B_4 = y_4[:,2]
print("Benzene Experiment 4")
NLL_B4 = NLL_models(eq_list_CB4,t,B_4,variance,NLL_B)
param_B4 = number_param('hall_of_fame_CB4.csv')
AIC_B4 = find_best_model(NLL_B4,param_B4)
print(AIC_B4)

M_4 = y_4[:,3]
print("Methane Experiment 4")
NLL_M4 = NLL_models(eq_list_CM4,t,M_4,variance,NLL_M)
param_M4 = number_param('hall_of_fame_CM4.csv')
AIC_M4 = find_best_model(NLL_M4,param_M4)
print(AIC_M4)


y_T_1 = eq_list_CT1[AIC_T1](t)
y_H_1 = eq_list_CH1[AIC_H1](t)
y_B_1 = eq_list_CB1[AIC_B1](t)
y_M_1 = eq_list_CM1[AIC_M1](t)

y_T_2 = eq_list_CT2[AIC_T2](t)
y_H_2 = eq_list_CH2[AIC_H2](t)
y_B_2 = eq_list_CB2[AIC_B2](t)
y_M_2 = eq_list_CM2[AIC_M2](t)

y_T_3 = eq_list_CT3[AIC_T3](t)
y_H_3 = eq_list_CH3[AIC_H3](t)
y_B_3 = eq_list_CB3[AIC_B3](t)
y_M_3 = eq_list_CM3[AIC_M3](t)

y_T_4 = eq_list_CT4[AIC_T4](t)
y_H_4 = eq_list_CH4[AIC_H4](t)
y_B_4 = eq_list_CB4[AIC_B4](t)
y_M_4 = eq_list_CM4[AIC_M4](t)


for i in range(initial_conditions.shape[1]):
# for i in range(2,3):
    fig, ax = plt.subplots()
    ax.set_title('Experiment '+exp_l[i])
    ax.set_ylabel('Concentrations $(M)$')
    ax.set_xlabel('Time $(s)$')
    if i == 0:
        ax.plot(t,y_T_1,color='royalblue',label='Toluene')
        ax.plot(t,y_H_1,color='lightskyblue',label='Hydrogen')
        ax.plot(t,y_B_1,color='lightsalmon',label='Benzene')
        ax.plot(t,y_M_1,color='tomato',label='Methane')
    if i == 1:
        ax.plot(t,y_T_2,color='royalblue',label='Toluene')
        ax.plot(t,y_H_2,color='lightskyblue',label='Hydrogen')
        ax.plot(t,y_B_2,color='lightsalmon',label='Benzene')
        ax.plot(t,y_M_2,color='tomato',label='Methane')
    if i == 2:
        ax.plot(t,y_T_3,color='royalblue',label='Toluene')
        ax.plot(t,y_H_3,color='lightskyblue',label='Hydrogen')
        ax.plot(t,y_B_3,color='lightsalmon',label='Benzene')
        ax.plot(t,y_M_3,color='tomato',label='Methane')
    if i == 3:
        ax.plot(t,y_T_4,color='royalblue',label='Toluene')
        ax.plot(t,y_H_4,color='lightskyblue',label='Hydrogen')
        ax.plot(t,y_B_4,color='lightsalmon',label='Benzene')
        ax.plot(t,y_M_4,color='tomato',label='Methane')
    # ax.plot(t,z[i][:,0],color='b',label='Toluene')
    # ax.plot(t,z[i][:,1],color='r',label='Hydrogen')
    # ax.plot(t,z[i][:,2],color='g',label='Methylcyclohexane')
    ax.plot(t,noisy_data[i][:,0],'o',markersize=2,color='b')
    ax.plot(t,noisy_data[i][:,1],'o',markersize=2,color='r')
    ax.plot(t,noisy_data[i][:,2],'o',markersize=2,color='g')
    ax.plot(t,noisy_data[i][:,3],'o',markersize=2,color='m')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.set_xticks(np.arange(0,51,10))
    # ax.set_yticks(np.arange(0,21,4))
    # ax.axis(xmin=0,xmax=51)
    # ax.axis(ymin=-1,ymax=21)
    ax.grid(alpha=0.5)
    ax.legend()
    name_fig = str('ConcProfile_'+exp_l[i]+'_HydroTol.png')
    plt.savefig(name_fig,dpi=600)

# plt.show()


"##################### Derivative Calculations #####################"

def true_derivatives(z,t):
    k_1 = 2 
    k_2 = 9 
    k_3 = 5
    dTdt = (-1)*((k_1*z[:,1]*z[:,0])/(1+k_2*z[:,2]+k_3*z[:,0]))
    dHdt = (-1)*((k_1*z[:,1]*z[:,0])/(1+k_2*z[:,2]+k_3*z[:,0]))
    dBdt = ((k_1*z[:,1]*z[:,0])/(1+k_2*z[:,2]+k_3*z[:,0]))
    dMdt = ((k_1*z[:,1]*z[:,0])/(1+k_2*z[:,2]+k_3*z[:,0]))
    dzdt = [dTdt,dHdt,dBdt,dMdt]
    return dzdt

true_trajectories = []
for i in range(initial_conditions.shape[1]):
    true_trajectories.append(true_derivatives(z[i],t))


T_1 = np.empty(len(t))
H_1 = np.empty(len(t))
B_1 = np.empty(len(t))
M_1 = np.empty(len(t))

T_2 = np.empty(len(t))
H_2 = np.empty(len(t))
B_2 = np.empty(len(t))
M_2 = np.empty(len(t))

T_3 = np.empty(len(t))
H_3 = np.empty(len(t))
B_3 = np.empty(len(t))
M_3 = np.empty(len(t))

T_4 = np.empty(len(t))
H_4 = np.empty(len(t))
B_4 = np.empty(len(t))
M_4 = np.empty(len(t))

for i in range(len(t)):
    T_1[i] = derivative(eq_list_CT1[AIC_T1],t[i],dx=1e-6)
    H_1[i] = derivative(eq_list_CH1[AIC_H1],t[i],dx=1e-6)
    B_1[i] = derivative(eq_list_CB1[AIC_B1],t[i],dx=1e-6)
    M_1[i] = derivative(eq_list_CM1[AIC_M1],t[i],dx=1e-6)
    
    T_2[i] = derivative(eq_list_CT2[AIC_T2],t[i],dx=1e-6)
    H_2[i] = derivative(eq_list_CH2[AIC_H2],t[i],dx=1e-6)
    B_2[i] = derivative(eq_list_CB2[AIC_B2],t[i],dx=1e-6)
    M_2[i] = derivative(eq_list_CM2[AIC_M2],t[i],dx=1e-6)

    T_3[i] = derivative(eq_list_CT3[AIC_T3],t[i],dx=1e-6)
    H_3[i] = derivative(eq_list_CH3[AIC_H3],t[i],dx=1e-6)
    B_3[i] = derivative(eq_list_CB3[AIC_B3],t[i],dx=1e-6)
    M_3[i] = derivative(eq_list_CM3[AIC_M3],t[i],dx=1e-6)

    T_4[i] = derivative(eq_list_CT4[AIC_T4],t[i],dx=1e-6)
    H_4[i] = derivative(eq_list_CH4[AIC_H4],t[i],dx=1e-6)
    B_4[i] = derivative(eq_list_CB4[AIC_B4],t[i],dx=1e-6)
    M_4[i] = derivative(eq_list_CM4[AIC_M4],t[i],dx=1e-6)

grand_derivative_T = np.concatenate([T_1,T_2,T_3,T_4])
grand_derivative_H = np.concatenate([H_1,H_2,H_3,H_4])
grand_derivative_B = np.concatenate([B_1,B_2,B_3,B_4])
grand_derivative_M = np.concatenate([M_1,M_2,M_3,M_4])


for i in range(initial_conditions.shape[1]):
    fig, ax = plt.subplots()
    ax.set_title('Experiment '+exp_l[i])
    ax.set_ylabel('$\dfrac{dC}{dt} \quad (Ms^{-1})$')
    ax.set_xlabel('Time $(s)$')
    if i == 0:
        ax.plot(t,true_trajectories[0][0],'o',markersize=3,color='b')
        ax.plot(t,true_trajectories[0][1],'x',markersize=3,color='r')
        ax.plot(t,true_trajectories[0][2],'+',markersize=3,color='g')
        ax.plot(t,true_trajectories[0][3],'^',markersize=3,color='m')
        ax.plot(t,T_1,label='Toluene',color='b')
        ax.plot(t,H_1,label='Hydrogen',color='r')
        ax.plot(t,B_1,label='Benzene',color='g')
        ax.plot(t,M_1,label='Methane',color='m')
        # ax.set_yticks(np.arange(-2,1,0.5))
        # ax.axis(ymin=-2,ymax=1)
    if i == 1:
        ax.plot(t,true_trajectories[1][0],'o',markersize=3,color='b')
        ax.plot(t,true_trajectories[1][1],'x',markersize=3,color='r')
        ax.plot(t,true_trajectories[1][2],'+',markersize=3,color='g')
        ax.plot(t,true_trajectories[1][3],'^',markersize=3,color='m')
        ax.plot(t,T_2,label='Toluene',color='b')
        ax.plot(t,H_2,label='Hydrogen',color='r')
        ax.plot(t,B_2,label='Benzene',color='g')
        ax.plot(t,M_2,label='Methane',color='m')
        # ax.set_yticks(np.arange(-2.5,0.5,0.5))
        # ax.axis(ymin=-2.5,ymax=0.5)
    if i == 2:
        ax.plot(t,true_trajectories[2][0],'o',markersize=3,color='b')
        ax.plot(t,true_trajectories[2][1],'x',markersize=3,color='r')
        ax.plot(t,true_trajectories[2][2],'+',markersize=3,color='g')
        ax.plot(t,true_trajectories[2][3],'^',markersize=3,color='m')
        ax.plot(t,T_3,label='Toluene',color='b')
        ax.plot(t,H_3,label='Hydrogen',color='r')
        ax.plot(t,B_3,label='Benzene',color='g')
        ax.plot(t,M_3,label='Methane',color='m')
        # ax.set_yticks(np.arange(-1,0.5,0.5))
        # ax.axis(ymin=-1,ymax=0.5)
    if i == 3:
        ax.plot(t,true_trajectories[3][0],'o',markersize=3,color='b')
        ax.plot(t,true_trajectories[3][1],'x',markersize=3,color='r')
        ax.plot(t,true_trajectories[3][2],'+',markersize=3,color='g')
        ax.plot(t,true_trajectories[3][3],'^',markersize=3,color='m')
        ax.plot(t,T_4,label='Toluene',color='b')
        ax.plot(t,H_4,label='Hydrogen',color='r')
        ax.plot(t,B_4,label='Benzene',color='g')
        ax.plot(t,M_4,label='Methane',color='m')
        # ax.set_yticks(np.arange(-1,0.5,0.5))
        # ax.axis(ymin=-1,ymax=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.set_xticks(np.arange(0,51,10))
    # ax.axis(xmin=-0.25,xmax=51)
    ax.grid(alpha=0.5)
    ax.legend()
    name_fig = str('Derivative_'+exp_l[i]+'_HydroTol.png')
    plt.savefig(name_fig,dpi=600)

# plt.show()


"################### Symbolic Regression Rate ###################"

# successful = False
# while successful is False:
#     try:
#         model = PySRRegressor(
#             niterations=200,
#             binary_operators=["+", "*", "/", "-"],
#             # unary_operators=["square","cube","sqrt"],
#             model_selection="accuracy",
#             loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
#             maxsize = 25,
#             timeout_in_seconds = 300,
#             parsimony = 1e-5,
#             equation_file= "hall_of_fame_rate.csv",
#             variable_names = ["T","H","B"],
#         )
        
# #         model.fit(grand_exp_data[:,:3],grand_derivative_T)
#         model.fit(grand_exp_data[:,:3],grand_derivative_T,variable_names=["T","H","B"])
#         successful = True
#     except:
#         pass


"##################### Optimise Rate Model #####################"

def rate_n_param(path):
    # read equations from CSV with different separator 
    # data = pd.read_csv(path,sep='|')
    data = pd.read_csv(path)
    # convert dataframe into numpy array
    eqs = data['Equation'].values
    T,H,B = symbols('T H B')
    # x0,x1,x2 = symbols('x0 x1 x2')
    simple_traj = []
    param = []
    for eq in eqs:
        func = simplify(eq)
        func = str(func)
        j = 0
        things = re.findall(r'(\*{2}|\*{0})(\d+\.?\d*)', func)
        for i in range(len(things)):
            if things[i][0] != '**':
                j += 1
        simple_traj.append(func)
        param.append(int(j))
    # simple_traj = np.array(simple_traj).tolist()
    return simple_traj, param


def best_rate_model(NLL,param):
    AIC = 2*np.array(NLL) + 2*np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]


def manipulate_equations(equations):
    rate = []
    for equation in equations:
        equation = str(equation)
        parameters = re.findall(r'(\*{2}|\*{0})(\d+\.?\d*)', equation)
        j = 0
        for i in range(len(parameters)):
            parameters[i] = list(parameters[i])
            if parameters[i][0] != '**':
                symbol = 'k' + str([j])
                j += 1
                equation = equation.replace(parameters[i][1],symbol,1)
        equation = equation.replace('T','z[0]')
        equation = equation.replace('H','z[1]')
        equation = equation.replace('B','z[2]')
        equation = equation.replace('sqrt_abs','np.sqrt')
        rate.append(equation)
    return rate


def ode(k,z0,equation):
    globals()["k"] = k
    def nest(z,t):
        dTdt = eval(str(equation))
        dHdt = eval(str(equation))
        dBdt = (-1)*eval(str(equation))
        dMdt = (-1)*eval(str(equation))
        dzdt = [dTdt, dHdt, dBdt, dMdt]
        return dzdt
    
    t = np.linspace(0,10,number_datapoints)
    z = odeint(nest,z0,t)
    
    return z


def NLL(inputs,model,variance,equation):
    number_exp = 4
    t = np.linspace(0,10,number_datapoints)
    likelihood = np.empty([number_exp,number_datapoints,4])
    
    for i in range(initial_conditions.shape[1]):
        z0 = initial_conditions[:,i]
        likelihood[i][:,0] = (((noisy_data[i][:,0] - model(inputs,z0,equation)[:,0])**2)/(2*(variance[0]**2))) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[0]**2))))
        likelihood[i][:,1] = (((noisy_data[i][:,1] - model(inputs,z0,equation)[:,1])**2)/(2*(variance[1]**2))) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[1]**2))))
        likelihood[i][:,2] = (((noisy_data[i][:,2] - model(inputs,z0,equation)[:,2])**2)/(2*(variance[2]**2))) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[2]**2))))
        likelihood[i][:,3] = (((noisy_data[i][:,3] - model(inputs,z0,equation)[:,3])**2)/(2*(variance[3]**2))) \
            - np.log(1/(np.sqrt(2*np.pi*(variance[3]**2))))


    return np.sum(likelihood) 


# def NLL(inputs,model,variance):
#     number_exp = 3
#     t = np.linspace(0,10,number_datapoints)
#     data_points = len(t)
#     likelihood = np.empty([number_exp,data_points,3])
    
#     for i in range(initial_conditions.shape[1]):
#         z0 = initial_conditions[:,i]
#         likelihood[i][:,0] = (((noisy_data[i][:,0] - model(inputs,z0)[:,0])**2)/(2*(variance[0]**2))) \
#             - np.log(1/(np.sqrt(2*np.pi*(variance[0]**2))))
#         likelihood[i][:,1] = (((noisy_data[i][:,1] - model(inputs,z0)[:,1])**2)/(2*(variance[1]**2))) \
#             - np.log(1/(np.sqrt(2*np.pi*(variance[1]**2))))
#         likelihood[i][:,2] = (((noisy_data[i][:,2] - model(inputs,z0)[:,2])**2)/(2*(variance[2]**2))) \
#             - np.log(1/(np.sqrt(2*np.pi*(variance[2]**2))))

#     return np.sum(likelihood) 


def Opt_Rout(multistart,number_parameters,lower_bound,upper_bound,to_opt,model,variance,equation):
    
    localsol = np.empty([multistart,number_parameters])
    localval = np.empty([multistart,1])
    boundss = Bounds(lower_bound,upper_bound)
    
    for i in range(multistart):
        x0 = np.random.uniform(lower_bound,upper_bound,size=number_parameters)
        res = minimize(to_opt, x0, args=(model,variance,equation), \
                        method='L-BFGS-B',bounds=boundss)
        localsol[i] = res.x
        localval[i] = res.fun

    minindex = np.argmin(localval)
    NLL = localval[minindex]
    opt_param = localsol[minindex]
    
    return NLL, opt_param


# def Opt_Rout(multistart,number_parameters,lower_bound,upper_bound,to_opt,model,variance):
    
#     localsol = np.empty([multistart,number_parameters])
#     localval = np.empty([multistart,1])
#     boundss = Bounds(lower_bound,upper_bound)
    
#     for i in range(multistart):
#         x0 = np.random.uniform(lower_bound,upper_bound,size=number_parameters)
#         res = minimize(to_opt, x0, args=(model,variance), \
#                         method='L-BFGS-B',bounds=boundss)
#         localsol[i] = res.x
#         localval[i] = res.fun

#     minindex = np.argmin(localval)
#     NLL = localval[minindex]
#     opt_param = localsol[minindex]
    
#     return NLL, opt_param


path = 'hall_of_fame_rate.csv'
simple_traj, param = rate_n_param(path)
rate = manipulate_equations(simple_traj)
print(simple_traj)
print(rate)
# rate = ['-k[0]', 
# '-k[0]*z[0]', 
# '-k[0]*z[1]*z[0]', 
# '-z[1]*z[0]/(z[2]*z[1] + k[0])', 
# '-k[0]*z[1]*z[0]/(z[2] + z[0])', 
# '-k[0]*z[1]*z[0]/(k[1]*z[2] + z[0])', 
# '-k[0]*z[1]*z[0]/(k[1]*z[2] + z[0] + k[2])', 
# '-k[0]*z[0]*(z[1] + k[1])/(k[2]*z[2] + z[0] + k[3])', 
# '-k[0]*z[0]*(z[1]**2 + k[1])/(z[1]*(k[2]*z[2] + z[0] + k[3]))', 
# '-k[0]*z[0]*(z[1]**2 + k[1])/(z[1]*(k[2]*z[2] + z[0] + k[3]))']

# mydict = {}
# NLL_rates = []
# variance = np.array([STD_T,STD_H,STD_B,STD_M])

# for i in range(len(rate)):
#     mydict[f'NLL_{i+1}, opt_param_{i+1}'] = Opt_Rout(5,param[i],0.0001,10,NLL,ode,variance,rate[i])
#     print(f'NLL_{i+1}, opt_param_{i+1}',mydict[f'NLL_{i+1}, opt_param_{i+1}'])
#     NLL_rates.append(mydict[f'NLL_{i+1}, opt_param_{i+1}'][0][0].astype(float))

# print(mydict)

# aaa = np.zeros(len(rate))
# for i in range(len(rate)):
#     key = 'NLL_' + str(i+1) + ', opt_param_' + str(i+1)
#     NLL_mod = mydict[key][0]
#     aaa[i] = 2*NLL_mod + 2*param[i]
#     print(rate[i],aaa[i])

# "################### Plot of Models ###################"

# def plot_opt(ode,opt_param,equation):
#     for i in range(initial_conditions.shape[1]):
#         fig, ax = plt.subplots()
#         ax.set_title('Experiment '+exp_l[i])
#         ax.set_ylabel('Concentration $(M)$')
#         ax.set_xlabel('Time $(t)$')
#         z0 = initial_conditions[:,i]
#         y2 = ode(opt_param,z0,equation)
#         ax.plot(t,noisy_data[i][:,0],'o',markersize=2,color='b')
#         ax.plot(t,noisy_data[i][:,1],'o',markersize=2,color='r')
#         ax.plot(t,noisy_data[i][:,2],'o',markersize=2,color='g')
#         ax.plot(t,y2[:,0], label = "Toluene", color='b')
#         ax.plot(t,y2[:,1], label = "Hydrogen", color='r')
#         ax.plot(t,y2[:,2], label = "Methylcyclohexane", color='g')
#         ax.spines["right"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         # ax.set_xticks(np.arange(0,51,10))
#         # ax.set_yticks(np.arange(0,21,4))
#         # ax.axis(xmin=0,xmax=51)
#         # ax.axis(ymin=-1,ymax=21)
#         ax.grid(alpha=0.5)
#         ax.legend()
#     plt.show()
#     return

# for i in range(len(rate)):
#     key = 'NLL_' + str(i+1) + ', opt_param_' + str(i+1)
#     opt_param = mydict[key][1]
#     plot_opt(ode,opt_param,rate[i])


# "##################### Model Based Design of Experiment|s #######################"

# def Diff(init_cond,model_1,param_1,model_2,param_2):
#     output_1 = model_1(param_1,init_cond)
#     output_2 = model_2(param_2,init_cond)
#     diff = (output_1-output_2)**2
#     difference = np.sum(diff)
#     return -difference
   
# def MBDoE(multistart,lower_bound_T,upper_bound_T,lower_bound_H,upper_bound_H,\
#           lower_bound_B,upper_bound_B,lower_bound_M,upper_bound_M,to_opt,\
#           model_1,param_1,model_2,param_2):
   
#     number_parameters = 4
#     localsol = np.empty([multistart,number_parameters])
#     localval = np.empty([multistart,1])
#     boundss = ((lower_bound_T,upper_bound_T),(lower_bound_H,upper_bound_H),\
#                 (lower_bound_B,upper_bound_B),(lower_bound_M,upper_bound_M))
#     # boundss = Bounds(lower_bound,upper_bound)
   
#     for i in range(multistart):
#         x0_T = np.random.uniform(lower_bound_T,upper_bound_T)
#         x0_H = np.random.uniform(lower_bound_H,upper_bound_H)
#         x0 = np.append(x0_T,x0_H)
#         x0_B = np.random.uniform(lower_bound_B,upper_bound_B)
#         x0 = np.append(x0,x0_B)
#         x0_M = np.random.uniform(lower_bound_M,upper_bound_M)
#         x0 = np.append(x0,x0_M)
#         res = minimize(to_opt, x0, args=(model_1,param_1,model_2,param_2), \
#                         method='L-BFGS-B',bounds=boundss)
#         localsol[i] = res.x
#         localval[i] = res.fun

#     minindex = np.argmin(localval)
#     next_exp = localsol[minindex]
   
#     return next_exp

# def competition_1(k,z0):

#     def competition_1_nest(z,t):
#         dTdt = (-z[1]*z[0]/(k[0]*z[2] + k[1]*z[1] - k[2]))
#         dHdt = (-z[1]*z[0]/(k[0]*z[2] + k[1]*z[1] - k[2]))
#         dBdt = (-1)*(-z[1]*z[0]/(k[0]*z[2] + k[1]*z[1] - k[2]))
#         dMdt = (-1)*(-z[1]*z[0]/(k[0]*z[2] + k[1]*z[1] - k[2]))
#         dzdt = [dTdt,dHdt,dBdt,dMdt]
#         return dzdt
        
#     # time points
#     t = np.linspace(0,10,number_datapoints)
    
#     # solve ODE
#     z = odeint(competition_1_nest,z0,t)
            
#     return z

# def competition_2(k,z0):

#     def competition_2_nest(z,t):
#         dTdt = (z[1]**2*z[0]/(z[1]*(-k[0]*z[2] - k[1]*z[1] + z[0] + k[2]) + z[0]))
#         dHdt = (z[1]**2*z[0]/(z[1]*(-k[0]*z[2] - k[1]*z[1] + z[0] + k[2]) + z[0]))
#         dBdt = (-1)*(z[1]**2*z[0]/(z[1]*(-k[0]*z[2] - k[1]*z[1] + z[0] + k[2]) + z[0]))
#         dMdt = (-1)*(z[1]**2*z[0]/(z[1]*(-k[0]*z[2] - k[1]*z[1] + z[0] + k[2]) + z[0]))
#         dzdt = [dTdt,dHdt,dBdt,dMdt]
#         return dzdt
        
#     # time points
#     t = np.linspace(0,10,number_datapoints)
    
#     # solve ODE
#     z = odeint(competition_2_nest,z0,t)
            
#     return z


# key = 'NLL_' + str(6+1) + ', opt_param_' + str(6+1)
# opt_param_1 = mydict[key][1]
# key = 'NLL_' + str(8+1) + ', opt_param_' + str(8+1)
# opt_param_2 = mydict[key][1]
# next_experiment = MBDoE(5,1,5,3,8,0,2,0.5,3,Diff,competition_1,opt_param_1,competition_2,opt_param_2)
# print('To discriminate between the 2 best models, experiment with given conditions:',next_experiment)


