# The Automated Discovery of Kinetic Rate Models - Methodological Frameworks

The industrialization of catalytic processes requires reliable kinetic models for their design, optimization and control. Mechanistic models require significant domain knowledge, while data-driven and hybrid models lack interpretability. Automated knowledge discovery methods, such as ALAMO, SINDy, and genetic programming, have gained popularity but suffer from limitations such as needing model structure assumptions, exhibiting poor scalability, and displaying sensitivity to noise. To overcome these challenges, we propose two methodological frameworks, ADoK-S and ADoK-W, for the automated generation of catalytic kinetic models using a robust criterion for model selection. We leverage genetic programming for model generation and a sequential optimization routine for model refinement. The frameworks are tested against three case studies of increasing complexity, demonstrating their ability to retrieve the underlying kinetic rate model with limited noisy data from the catalytic systems, showcasing their potential for chemical reaction engineering applications.

## Notation

Here we also set the necessary mathematical notation to describe our methods precisely.
We start from the standard symbolic regression formulation to later introduce the weak and strong variations of our framework.

The domain set $\mathcal{Z}$ is the union of an arbitrary number of constants $\Gamma$ and a fixed number of variables $\mathcal{X}$.
The operator set $\mathcal{P}$ is the union of arithmetic operations ($\diamond: \mathbb{R}^2 \rightarrow \mathbb{R}$) and a finite set of special one-dimensional functions ($\Lambda: \mathbb{R} \rightarrow \mathbb{R}$).
The model search space $\mathcal{M}$ is the space of possible expressions to be reached by iterative function composition of the operator set $\mathcal{P}$ over the domain set $\mathcal{Z}$.

The variables can be represented as state vectors $x \in \mathbb{R}^{n_x}$.
A data point is a pair of specific states $x$ and the associated target value $y \in \mathbb{R}$ of an unknown function $f: \mathbb{R}^{n_x} \rightarrow \mathbb{R}$: $y=f(x)$.
The data set $\mathcal{D}$ consists of $n_t$ data points: $\mathcal{D} = \lbrace(x^{(i)}, y^{(i)}) \mid  i = 1, \ldots, n_t \rbrace$.
To quantify the discrepancy between the predictions and the target values, we can leverage any adequate positive measure function $\ell:\mathbb{R}^{n} \times \mathbb{R}^{n} \rightarrow \mathbb{R}^+$.

A symbolic model $m \in \mathcal{M}$ has a finite set of parameters $\theta_m$ whose dimension $d_m$ depends on the model.
We denote the prediction of a model under specific parameter values in functional form as $m(\cdot\mid\theta_m)$.
We use $\hat{y}_m$ to denote the prediction of a value coming from a proposed model $m$ (i.e., $\hat{y}_m = m(\cdot \mid \theta_m) $).

For our purposes, it is important to decouple the model generation step from the parameter optimization for each model.
An optimal model $m^*$ is defined as

$$m^*=\arg\min_{m \in \mathcal{M}} \sum_{i=1}^{n_t}{\ell\left(\hat{y}_m^{(i)}, y^{(i)}\right)},$$

and its optimal parameters are such that

$$\theta_{m^\*}^\* = \arg\min_{\theta_{m^\*}^\*} \sum_{i=1}^{n_t}{\ell\left(\hat{y}_{m^\*}^{(i)}, y^{(i)}\right)}.$$

In dynamical systems, the state variables are a function of time, $x(t) \in \mathbb{R}^{n_x}$, and represent the evolution of the real dynamical system within a fixed time interval $\Delta t = [t_0, t_f]$.
The dynamics are defined by the rates of change $\dot{x}(t) \in \mathbb{R}^{n_x}$ and the initial condition $x_0 = x(t=t_0)$.

For our kinetic rate models, we assume that the $n_t$ sampling times are set within the fixed time interval, $t^{(i)} \in \Delta t$.
The concentration measurements $C$ at each time point $t^{(i)}$ are samples of the real evolution of the system $C^{(i)} \approx x(t^{(i)})$, while the rate estimates $r$ are an approximation to the rate of change $r^{(i)} \approx \dot{x}(t^{(i)})$.

Here the available data set $\mathcal{D}$ is formed by ordered pairs of time and state measurements $\mathcal{D} = \{(t^{(i)}, C^{(i)})\mid i = 1,\ldots, n_t\}$.
As before, we use a hat to denote the prediction of either states $\hat{C}_m$ or rates $\hat{r}_m$ coming from a proposed model $m$.
The output of the models with specific parameters $\theta_m$ are denoted as $\hat{C}_m(\cdot \mid \theta_m)$ and  $\hat{r}_m(\cdot\mid\theta_m)$, respectively.

The complexity of a model is denoted as $\mathcal{C}(m)$ (here we use the number of nodes in an expression tree as the complexity of a symbolic expression).
We distinguish between families of expressions with different levels of complexity $\kappa \in \mathbb{N}$ as $\mathcal{M}^\kappa = \lbrace m \in \mathcal{M} \mid \mathcal{C}(m) = \kappa \rbrace$.

## Introduction to ADoK-S
For ADoK-S, the objective is to find the model $m$ that best maps the states to the rates:

$$\hat{r}_m(t \mid \theta_m) = m(x(t) \mid \theta_m).$$

For this to be done directly, an estimation of the rates of change $r^{(i)}$ must be derived from the available concentration measurements $C^{(i)}$. To solve this, our approach forms an intermediate symbolic model $\eta$ such that $\eta(t^{(i)}) \approx C^{(i)}$ following the standard symbolic regression procedure, described in the equations presented above, with our model selection process described below.

Since this model is differentiable, its derivatives provide an approximation to the desired rates: $\dot{\eta}\left(t^{(i)}\right) \approx r^{(i)}$. With these estimated values available, the optimization problem can be written as follows. The outer level optimizes over model proposals for a fixed level of complexity $\kappa$,

$$m^\star = \arg\min_{m \in \mathcal{M^\kappa}} \sum_{i=1}^{n_t} \ell\left(\hat r_m(t^{(i)}\mid\theta_m), r^{(i)} \right),$$

while the inner level optimizes over the best model's parameters,

$$\theta_{m^\star}^\star = \arg\min_{\theta_{m^\star}} \sum_{i=1}^{n_t} \ell\left(\hat r_{m^\star}(t^{(i)}\mid\theta_{m^\star}), r^{(i)} \right).$$

In the above equations, $\ell$ represents the residual sum of squares (RSS).

## Introduction to ADoK-W
For ADoK-W, we aim to find the model $m$ that best maps state variables to the differential equation system that define the state dynamics to then predict the concentration evolution:

$$\dot x_m(t \mid \theta_m) = m(x(t)\mid \theta_m),$$

$$\hat C_m(t\mid \theta_m) = C_0 + \int_{t_0}^{t} \dot x_m(\tau \mid \theta_m) d\tau,$$

where the initial condition $C_0$ is the first concentration measurement. For this formulation, the outer level optimizes over model proposals for a specific complexity level $\kappa$ as well,

$$m^\star = \arg\min_{m \in \mathcal{M^\kappa}} \sum_{i=1}^{n_t} \ell\left(\hat C_m(t^{(i)}\mid\theta_m), C^{(i)} \right),$$

while the inner level optimizes over the parameters of the best model,

$$\theta_{m^\star}^\star = \arg\min_{\theta_{m^\star}} \sum_{i=1}^{n_t} \ell\left(\hat C_{m^\star}(t^{(i)}\mid\theta_{m^\star}), C^{(i)} \right).$$

Again, in the above equations, $\ell$ represents the RSS.

## Model Selection
Given a model $m$ with parameters $\theta_m$ of dimension $d_m$, the Akaike information criterion (AIC) is defined as

$$\text{AIC}_m = 2\mathcal{L}(\mathbf{\theta}_m\mid\mathcal{D}) + 2d_m,$$

where $\mathcal{L}$ represents specifically the negative log-likelihood (NLL). Given two competing models, $m_1$ and $m_2$, the preferred model would be the one with the lowest AIC value calculated by presented equation. The choice of AIC for model selection within the ADoK-S and ADoK-W framework is motivated in detail in the full paper.

## Model-Based Design of Experiments
It is possible that the data set used for the regression is not enough to provide an adequate model proposal. For this scenario, and under the assumption that the experimental budget is not fully spent, it is possible to leverage the implicit insights in the optimized models to extract an informative proposal for a new experiment. For this purpose, we may search for an initial condition which maximizes the discrepancy between state predictions $\hat x(t)$ of the best two proposed models, $\eta$ and $\mu$, using the available data set:

$$x_0^{(new)} = \arg\max_{x_0} \int_{t_0}^{t_f} \ell\left(\hat x_\eta \left(\tau\mid\theta_\eta^\star \right), \hat x_\mu \left(\tau\mid\theta_\mu^\star \right) \right)\, d\tau.$$

In the above equation, $\ell$ represents the RSS. Starting from the proposed initial condition, an experiment can be carried out to obtain a new batch of data points to be added to the original data set. Finally, the whole process of model proposal and selection can be redone with the enhanced data set, closing the loop between informative experiments and optimal models.

## Code Example for ADoK-S
### Import Important Packages
Below we show the needed packages to install for the rest of the example to run.

```ruby
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
```

### Data Generation
Here, we will be working with the hydrodealkylation of toluene as a case study. The first thing that we must do is generate some data (if experimental data is not available - if it is, it should be formatted in the same way it is presented above).

```ruby
def kinetic_model(t, z):
    # Define rate constants for the kinetic model
    k_1 = 2 
    k_2 = 9 
    k_3 = 5

    # Define the differential equations for the kinetic model
    dTdt = (-1) * ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dHdt = (-1) * ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dBdt = ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dMdt = ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))

    # Return the derivatives as a list
    dzdt = [dTdt, dHdt, dBdt, dMdt]
    return dzdt

# List of species involved in the kinetic model
species = ['T', 'H', 'B', 'M']

# Define initial conditions for each experiment
initial_conditions = {
    "ic_1": np.array([1, 8, 2, 3]),
    "ic_2": np.array([5, 8, 0, 0.5]),
    "ic_3": np.array([5, 3, 0, 0.5]),
    "ic_4": np.array([1, 3, 0, 3]),
    "ic_5": np.array([1, 8, 2, 0.5])
}

# Calculate the number of experiments and species
num_exp = len(initial_conditions)
num_species = len(species)

# Define time parameters for the simulation
timesteps = 30
time = np.linspace(0, 10, timesteps)
t = [0, np.max(time)]
t_eval = list(time)

# Define standard deviation for noise
STD = 0.2

# Generate synthetic noise for each experiment
noise = [np.random.normal(0, STD, size = (num_species, timesteps)) for i in range(num_exp)]

# Dictionaries to store simulated data
in_silico_data = {}
no_noise_data = {}

# Simulate each experiment and add noise to the data
for i in range(num_exp):
    ic = initial_conditions["ic_" + str(i + 1)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    in_silico_data["exp_" + str(i + 1)] = np.clip(solution.y + noise[i], 0, 1e99)
    no_noise_data["exp_" + str(i + 1)] = solution.y

# Plotting the in-silico data for visualisation
for i in range(num_exp):
    fig, ax = plt.subplots()
    ax.set_title("Experiment " + str(i + 1))
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

plt.show()
```

### Generating Concentration Models
Once we have generated the concentration versus time dataset, we must now create concentration profiles so we can then numerically differentiate them and approximate the rates of reaction (which cannot be measured experimentally). The inputs for the genetic programming algorithm can be changed in accordance to one's problems. This snippet of code will generate files with the equations 

```ruby
# Loop through each experiment and species to perform symbolic regression
for i in range(num_exp):
    for j in range(num_species):
        successful = False  # Flag to indicate successful completion

        # Repeat until successful
        while successful is False:
            try:
                # Selecting file names based on species for saving results
                if j == 0:
                    file_name = str("hall_of_fame_T_" + str(i + 1) + ".csv")
                elif j == 1:
                    file_name = str("hall_of_fame_H_" + str(i + 1) + ".csv")
                elif j == 2:
                    file_name = str("hall_of_fame_B_" + str(i + 1) + ".csv")
                elif j == 3:
                    file_name = str("hall_of_fame_M_" + str(i + 1) + ".csv")

                # Preparing time (X) and concentration (Y) data for regression
                X = time.reshape(-1, 1)
                Y = in_silico_data["exp_" + str(i + 1)][j].reshape(-1, 1)

                # Setting up the symbolic regression model with specified parameters
                model = PySRRegressor(
                    niterations=200,
                    binary_operators=["+", "*", "/", "-"],
                    unary_operators=["exp"],
                    model_selection="accuracy",
                    loss="loss(x, y) = (x - y)^2",
                    maxsize=9,
                    timeout_in_seconds=300,
                    parsimony=1e-5,
                    equation_file=file_name
                )

                # Fitting the model to the data
                model.fit(X, Y, variable_names=["t"])

                # Mark as successful to exit the while loop
                successful = True
            except:
                # If an exception occurs, the loop continues
                pass
```

### Finding the Best Concentration Models
Once the concentration models have been produced, we will read them from the files that we generated using the snippet above. We will need to evaluate the models generated in order for us to select the ones that minimize the AIC value. This can be done with the following code.

```ruby
def read_equations(path):
    # Reads equations from a CSV file
    data = pd.read_csv(path)
    # Retrieves the "Equation" column as a numpy array
    eqs = data["Equation"].values
    
    eq_list = []
    
    def make_f(eq):
        # Converts a string equation into a function
        def f(t):
            # Replace the variable in the equation with 't' and convert 'exp' to its numpy equivalent
            equation = eq.replace("x0", "t")
            return eval(equation.replace("exp", "np.exp"))
        return f
    
    for eq in eqs:
        # Convert each equation in the list into a function and add it to eq_list
        eq_list += [make_f(eq)]
    
    return eq_list

def number_param(path):
    # Reads equations from a CSV file
    data = pd.read_csv(path)
    eqs = data["Equation"].values
    t = symbols("t")
    simple_traj = []
    param = []
    
    for eq in eqs:
        # Simplifies each equation and counts the number of parameters
        func = simplify(eq)
        simple_traj.append(func)
        things = list(func.atoms(Float))
        param.append(len(things))
    
    simple_traj = np.array(simple_traj).tolist()
    return param

def find_best_model(NLL, param):
    # Calculates the AIC for each model and identifies the one with the lowest AIC
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def NLL_models(eq_list, t, data, NLL_species, number_datapoints):
    # Computes the Negative Log-Likelihood for each model
    NLL = []
    
    for f in eq_list:
        y_T = []
        
        for a in t:
            y_T.append(f(a))
        
        NLL.append(NLL_species(data, y_T, number_datapoints))
    return NLL

def NLL(C, y_C, number_datapoints):
    # Calculates the Negative Log-Likelihood for a given set of data points
    likelihood = np.empty(number_datapoints)
    mse = np.empty(number_datapoints)
    
    for i in range(number_datapoints):
        mse[i] = ((C[i] - y_C[i])**2)
    
    variance = np.sum(mse) / number_datapoints
    
    for i in range(number_datapoints):
        likelihood[i] = ((C[i] - y_C[i])**2) / (2 * variance) - np.log(1 / (np.sqrt(2 * np.pi * variance)))
    
    return np.sum(likelihood)

# Identifying the best concentration models for each experiment
equation_lists = {}
best_models = {}

for i in range(num_exp):
    data = in_silico_data["exp_" + str(i + 1)]
    
    for j in range(num_species):
        # Determine the file name based on the species and experiment number
        # Read equations, calculate NLL, and find the best model
        # File naming follows a specific pattern based on species and experiment number
        # The best model is identified for each species in each experiment

        if j == 0:
            file_name = str("hall_of_fame_T_" + str(i + 1) + ".csv")
            name = "T_"
        if j == 1:
            file_name = str("hall_of_fame_H_" + str(i + 1) + ".csv")
            name = "H_"
        if j == 2:
            file_name = str("hall_of_fame_B_" + str(i + 1) + ".csv")
            name = "B_"
        if j == 3:
            file_name = str("hall_of_fame_M_" + str(i + 1) + ".csv")
            name = "M_"

        # Read equations, calculate NLL values, and find the best model
        a = read_equations(file_name)
        nll_a = NLL_models(a, time, data[j], NLL, timesteps)
        param_a = number_param(file_name)
        best_models[name + str(i + 1)] = find_best_model(nll_a, param_a)
        equation_lists[name + str(i + 1)] = a

# Plotting the concentration profiles and in-silico data for each experiment
for i in range(num_exp):
    fig, ax = plt.subplots()
    ax.set_ylabel("Concentrations $(M)$", fontsize=18)
    ax.set_xlabel("Time $(h)$", fontsize=18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    for j in range(num_species):
        # Plot the actual data and the model predictions for each species
        y = in_silico_data["exp_" + str(i + 1)][j]
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        yy = equation_lists[name][model](time)
        ax.plot(time, y, marker[j], markersize=4, label=species[j], color=color_1[j])
        ax.plot(time, yy, color=color_1[j], linestyle="-")
    
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right', fontsize=15)
    
plt.show()
```

### Numerically Differentiating the Best Concentration Models
Now that we have figured out which concentration models minimize the AIC (and we have plotted the models versus the in-silico data to ensure that the models are capturing the trends of the kinetic data), we must differentiate our models so that we can approximate the rate measurements that we do not have direct access to. Since we are working with a synthetic dataset, we will also plot the approximations to the true rate dataset.

```ruby
derivatives = {}
SR_derivatives_T = np.array([])
SR_derivatives_H = np.array([])
SR_derivatives_B = np.array([])
SR_derivatives_M = np.array([])

# Calculate and store derivatives of concentration models for each experiment and species
for i in range(num_exp):
    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        best_model = equation_lists[name][model]
        derivative = np.zeros(timesteps)
        
        # Numerically differentiate the best model for each time step
        for h in range(timesteps):
            derivative[h] = der(best_model, time[h], dx=1e-6)
        
        # Store the derivative in a dictionary
        derivatives[name] = derivative

# Plotting the estimated and actual rates for each experiment
for i in range(num_exp):
    fig, ax = plt.subplots()
    ax.set_ylabel("Rate $(Mh^{-1})$", fontsize=18)
    ax.set_xlabel("Time $(h)$", fontsize=18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    data = no_noise_data["exp_" + str(i + 1)]
    y = kinetic_model(time, data)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        yy = derivatives[name]
        # Plot actual and estimated rates
        ax.plot(time, y[j], marker[j], markersize=4, label=species[j], color=color_1[j])
        ax.plot(time, yy, color=color_1[j], linestyle="-")
    
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right', fontsize=15)
plt.show()

# Concatenate derivatives for all experiments for each species
# to prepare for the second step of the symbolic regression methodology
for i in range(num_exp):
    SR_derivatives_T = np.concatenate([SR_derivatives_T, derivatives["T_" + str(i + 1)]])
    SR_derivatives_H = np.concatenate([SR_derivatives_H, derivatives["H_" + str(i + 1)]])
    SR_derivatives_B = np.concatenate([SR_derivatives_B, derivatives["B_" + str(i + 1)]])
    SR_derivatives_M = np.concatenate([SR_derivatives_M, derivatives["M_" + str(i + 1)]])

# Stacking data from different experiments for symbolic regression analysis
a = in_silico_data["exp_1"].T
b = in_silico_data["exp_2"].T
SR_data = np.vstack((a, b))
```

### Generate Rate Models
So far we have: (i) generated some kinetic data; (ii) using the kinetic data, construct concentration models for each species in each experiment; (iii) based on the constructed concentration models, we selected the best one based on AIC; (iv) from the best concentration model, we numerically differentiate it to approximate the rate of consumption and generation of the species. Now, with the approximations, we can use them to make rate models and again select the best rate model from the generated files. Below, using the genetic programming package, we make the rate models and save them as csv files (in the process, a bkup and a pickle file will be generated in the same directory, but these will not be used at all).

```ruby
# Loop over each species to perform symbolic regression for rate models
for i in range(num_species):
    successful = False  # Flag to indicate successful completion of symbolic regression

    # Repeat until successful
    while is successful False:
        try:
            # Selecting file names and corresponding derivative data based on species
            if i == 0:
                file_name = "hall_of_fame_rate_T.csv"
                Y = SR_derivatives_T
                num = 2000  # Number of iterations for the regression algorithm
            if i == 1:
                file_name = "hall_of_fame_rate_H.csv"
                Y = SR_derivatives_H
                num = 2000
            if i == 2:
                file_name = "hall_of_fame_rate_B.csv"
                Y = SR_derivatives_B
                num = 2000
            if i == 3:
                file_name = "hall_of_fame_rate_M.csv"
                Y = SR_derivatives_M
                num = 2000

            # Setting up the symbolic regression model with specified parameters
            model = PySRRegressor(
                niterations=num,
                binary_operators=["+", "*", "/", "-"],
                model_selection="accuracy",
                loss="loss(x, y) = (x - y)^2",
                maxsize=25,
                timeout_in_seconds=300,
                parsimony=1e-5,
                equation_file=file_name
            )

            # Fitting the model to the data
            # Using concentration data as input variables for the model
            model.fit(SR_data[:, 0:3].reshape(-1, 3), Y, variable_names=["T", "H", "B"])

            # Mark as successful to exit the while loop
            successful = True
        except:
            # If an exception occurs, the loop continues
            pass
```

### Selecting the Best Rate Model Generated
Similarly to what was done with the concentration models, we need to evaluate the generated rate models and find which one minimizes the AIC.

```ruby
def rate_n_param(path):
    # Read equations from a CSV file and extract the number of parameters in each equation
    data = pd.read_csv(path)
    eqs = data["Equation"].values
    T, H, B, M = symbols("T H B M")  # Define symbolic variables
    simple_traj = []  # List to store simplified equations
    param = []  # List to store the number of parameters for each equation
    
    for eq in eqs:
        func = simplify(eq)  # Simplify the equation
        func = str(func)  # Convert the simplified equation to a string
        j = 0
        # Find all numbers and operators in the equation
        things = re.findall(r"(\*{2}|\*{0})(\d+\.?\d*)", func)
        
        # Count the number of parameters
        for i in range(len(things)):
            if things[i][0] != "**":
                j += 1
        
        simple_traj.append(func)  # Add the simplified equation to the list
        param.append(int(j))  # Add the number of parameters to the list
    
    return simple_traj, param

# Dictionary to store rate models and their parameters
rate_models = {}
GP_models = {}

# Loop to read equations and their parameters for each species
for i in range(num_species):
    # Define the path and names for storing models and parameters
    # based on species index

    if i == 0:
        path = "hall_of_fame_rate_T.csv"
        name_models = "T_models"
        name_params = "T_params"
    
    if i == 1:
        path = "hall_of_fame_rate_H.csv"
        name_models = "H_models"
        name_params = "H_params"
    
    if i == 2:
        path = "hall_of_fame_rate_B.csv"
        name_models = "B_models"
        name_params = "B_params"
    
    if i == 3:
        path = "hall_of_fame_rate_M.csv"
        name_models = "M_models"
        name_params = "M_params"
    
    # Read equations and their parameters
    a, b = rate_n_param(path)
    # Store the equations and parameters in the dictionary
    GP_models[name_models, name_params] = a, b

def rate_model(z0, equations, t, t_eval, event):
    # Function to solve the ODE system given initial conditions, equations, and time steps
    i = 0
    
    # Replace symbols in equations with actual variable names
    for equation in equations:
        equation = str(equation)
        equation = equation.replace("T", "z[0]")
        equation = equation.replace("H", "z[1]")
        equation = equation.replace("B", "z[2]")
        equation = equation.replace("M", "z[3]")
        equations[i] = equation
        i += 1
    
    # Define the ODE system
    def nest(t, z):
        dTdt = eval(str(equations[0]))
        dHdt = eval(str(equations[0]))
        dBdt = (-1) * eval(str(equations[0]))
        dMdt = (-1) * eval(str(equations[0]))
        dzdt = [dTdt, dHdt, dBdt, dMdt]
        return dzdt
    
    # Solve the ODE system
    sol = solve_ivp(nest, t, z0, t_eval=t_eval, method="RK45", events=event)  
    
    return sol.y, sol.t, sol.status

# Initializing variables and arrays for storing equations and parameters
equations = []
names = ["T_models", "T_params", "H_models", "H_params", "B_models", "B_params", "M_models", "M_params"]
all_models = []  # To store all models for each species
params = []  # To store the number of parameters for each model

# Create all possible ODEs and count their parameters
for i in np.arange(0, len(names), 2):
    all_models.append(GP_models[names[i], names[i + 1]][0])
    params.append(GP_models[names[i], names[i + 1]][1])

# Function to calculate the Negative Log-Likelihood (NLL) for a given ODE system
def NLL_kinetics(experiments, predictions, number_species, number_datapoints):
    # Initialize arrays for calculations
    output = np.zeros(number_species)
    mse = np.zeros(number_species)
    variance = np.zeros(number_species)
    
    # Calculate MSE and variance for each species
    for i in range(number_species):
        mse[i] = np.sum((experiments[i] - predictions[i])**2)
        variance[i] = mse[i] / number_datapoints
    
    # Calculate the NLL for each species
    for i in range(number_species):
        likelihood = ((experiments[i] - predictions[i])**2) / (2 * variance[i]) \
            - np.log(1 / (np.sqrt(2 * np.pi * variance[i])))
        output[i] = np.sum(likelihood)
    
    return np.sum(output)

# Event function for solve_ivp to handle timeout
def my_event(t, y):
    time_out = perf_counter()
    return 0 if (time_out - time_in) > 5 else 1
my_event.terminal = True

all_ODEs = GP_models["H_models", "H_params"][0]
number_models = len(all_ODEs)
all_ODEs = [[x] for x in all_ODEs]
AIC_values = np.zeros(number_models)

# Evaluate NLL for each ODE model across all experiments
for i in range(number_models):
    neg_log = 0
    print(i)
    
    for j in range(num_exp):
        t = time
        experiments = in_silico_data["exp_" + str(j + 1)]
        time_in = perf_counter()
        ics = initial_conditions["ic_" + str(j + 1)]
        y, tt, status = rate_model(ics, list(all_ODEs[i]), [0, np.max(t)], list(t), my_event)
        
        # Assign a high penalty if the model fails to solve or takes too long
        if status in [-1, 1]:
            neg_log = 1e99
            break
        else:
            neg_log += NLL_kinetics(experiments, y, num_species, timesteps)
    
    # Calculate AIC values
    num_parameters = np.sum(np.array(params[1][i]))
    AIC_values[i] = 2 * neg_log + 2 * num_parameters

# Identifying the best model based on AIC values
best_model_index = np.argmin(AIC_values)
second_min_index = np.argpartition(AIC_values, 1)[1]

# Plot the best model's predictions against the actual data for each experiment
for i in range(num_exp):
    t = time
    time_in = perf_counter()
    ics = initial_conditions["ic_" + str(j + 1)]
    yy, tt, _ = rate_model(ics, list(all_ODEs[best_model_index]), [0, np.max(t)], list(t), my_event)
    
    # Set up plot
    fig, ax = plt.subplots()
    ax.set_ylabel("Concentrations $(M)$", fontsize=18)
    ax.set_xlabel("Time $(h)$", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot data for each species
    for j in range(num_species):
        y = in_silico_data["exp_" + str(j + 1)][j]
        ax.plot(t, y, "o", markersize=4, label=species[j], color=color_1[j])
        ax.plot(tt, yy[j], color=color_1[j])
    
    # Finalize plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right', fontsize=15)

plt.show()

# Print the best and second-best model equations
print(all_ODEs[best_model_index])
print(all_ODEs[second_min_index])
```


## Citation
```ruby
@misc{https://doi.org/10.48550/arxiv.2301.11356,
  doi = {10.48550/ARXIV.2301.11356},
  author = {de Carvalho Servia,  Miguel Ángel and Sandoval,  Ilya Orson and Hellgardt,  Klaus and Hii,  King Kuok and Zhang,  Dongda and del Rio Chanona,  Ehecatl Antonio},
  title = {{The Automated Discovery of Kinetic Rate Models -- Methodological Frameworks}},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
