"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

from pprint import PrettyPrinter

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config
from jaxutils import Dataset
import jaxkern as jk
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from scipy.optimize import minimize

import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
pp = PrettyPrinter(indent = 4)
key = jr.PRNGKey(123)

np.random.seed(1998)


"##############################################################################"
"############### Kinetic model for hydrodealkylation of toluene ###############"
"##############################################################################"

def kinetic_model(t, z):
    k_1 = 2 
    k_2 = 5

    dNOdt = (-1) * ((k_1 * z[0]**2) / (1 + k_2 * z[0]))
    dNdt = ((k_1 * z[0]**2) / (1 + k_2 * z[0]))
    dOdt = (1/2) * ((k_1 * z[0]**2) / (1 + k_2 * z[0]))

    dzdt = [dNOdt, dNdt, dOdt]
    return dzdt

species = ["NO", "N", "O"]
num_species = len(species)

initial_conditions = {
    "ic_1": np.array([5 , 0, 0]),
    "ic_2": np.array([10, 0, 0]),
    "ic_3": np.array([5 , 2, 0]),
    "ic_4": np.array([5 , 0, 3]),
    "ic_5": np.array([0 , 2, 3]),
}

number_exp = len(initial_conditions)
timesteps = 30
time = np.linspace(0, 10, timesteps)
t = [0, np.max(time)]
t_eval = list(time)

training_data = np.empty((timesteps * number_exp, num_species))
in_silico_data = {}
color_1 = cm.viridis(np.linspace(0, 1, num_species))
marker = ['o', '+', 'x', 'v']
stds = np.array([0., 0., 0.])


for i in range(number_exp):
    ic = initial_conditions['ic_' + str(i + 1)]
    nois = [np.random.normal(0, stds[j], size = (timesteps)) for j in range(num_species)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    solution = np.transpose(solution.y)
    noise = np.transpose(nois)
    in_silico_data['exp_' + str(i + 1)] = np.clip(solution + noise, 0, 1e99).reshape(-1, num_species)
    beg = i * timesteps
    end = ((i+1) * timesteps)
    training_data[beg:end] = in_silico_data['exp_' + str(i + 1)]

for i in range(number_exp):
    fig, ax = plt.subplots()
    ax.set_title('Experiment ' + str(i + 1))
    ax.set_ylabel('Concentrations $(M)$')
    ax.set_xlabel('Time $(h)$')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for j in range(num_species):
        y = in_silico_data['exp_' + str(i + 1)][:, j]
        ax.plot(time, y, marker[j], markersize = 3, label = species[j], color = color_1[j])

    ax.grid(alpha = 0.5)
    ax.legend()


"##############################################################################"
"########################### Pre-treatment of data ############################"
"##############################################################################"

def normalise(x):
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    normal = (x - mean) / std
    return normal, mean, std

normalised_data, data_mean, data_std = normalise(training_data)
input_training = np.empty((number_exp * (timesteps - 1), num_species))
output_training = np.empty((number_exp * (timesteps - 1), num_species))

for i in range(number_exp):
    beg = i * timesteps
    end = ((i+1) * timesteps)
    data = normalised_data[beg:end]
    beg = i * (timesteps - 1)
    end = ((i+1) * (timesteps - 1))
    input_training[beg:end] = np.delete(data, (timesteps - 1), axis = 0)
    output_training[beg:end] = np.delete(data, 0, axis = 0)

output_training_data = {}
D = {}

for i in range(num_species):
    output_training_data[species[i]] = output_training[:, i].reshape(-1, 1)
    D[species[i]] = Dataset(X = input_training, y = output_training_data[species[i]])


"##############################################################################"
"########################## Training GP and inference #########################"
"##############################################################################"

kernel = {}
prior = {}
likelihood = {}
posterior = {}
parameter_state = {}
params = {}
negative_mll = {}

for i in range(num_species):
    kernel[species[i]] = jk.RBF()
    prior[species[i]] = gpx.Prior(kernel = kernel[species[i]])
    likelihood[species[i]] = gpx.Gaussian(num_datapoints = D[species[i]].n)
    posterior[species[i]] = prior[species[i]] * likelihood[species[i]]
    parameter_state[species[i]] = gpx.initialise(
        posterior[species[i]], key, kernel = {"lengthscale": jnp.array([1])}
    )
    params[species[i]], trainable, bijectors = parameter_state[species[i]].unpack()
    negative_mll[species[i]] = jit(
        posterior[species[i]].marginal_log_likelihood(D[species[i]], negative = True)
    )
    negative_mll[species[i]](params[species[i]])


optimiser = ox.adam(learning_rate = 0.01)
inference_state = {}
learned_params = {}

for i in range(num_species):
    inference_state[species[i]] = gpx.fit(
        objective = negative_mll[species[i]],
        parameter_state = parameter_state[species[i]],
        optax_optim = optimiser,
        num_iters = 3000,
    )
    learned_params[species[i]], training_history = inference_state[species[i]].unpack()


ics = np.array([initial_conditions["ic_" + str(i + 1)] for i in range(number_exp)])
min_C = np.min(ics, axis = 0)
max_C = np.max(ics, axis = 0)
test_ic = np.empty(num_species)

for i in range(num_species):
    test_ic[i] = np.random.uniform(min_C[i], max_C[i])

norm_ic_test = (test_ic - data_mean) / data_std
new_input = norm_ic_test.reshape(-1, num_species)

latent_dist = {}
predictive_dist = {}
predictive_mean = {}
predictive_std = {}
predictions = np.empty((timesteps, num_species))
variances = np.empty((timesteps, num_species))

for i in range(timesteps):
    predictions[i] = new_input.reshape(num_species)

    for j in range(num_species):
        latent_dist[species[j]] = posterior[species[j]](
            learned_params[species[j]], D[species[j]])(new_input)
        predictive_dist[species[j]] = likelihood[species[j]](
            learned_params[species[j]], latent_dist[species[j]]
        )
        predictive_mean[species[j]] = predictive_dist[species[j]].mean()
        predictive_std[species[j]] = predictive_dist[species[j]].stddev()
        new_input[0][j] = predictive_mean[species[j]]

        if i == 0:
            variances[i][j] = 0
        
        else:
            variances[i][j] = predictive_std[species[j]]

def unnormalise(x, mean_x, std_x):
    return (x * std_x) + mean_x

unnormalised_predictions = unnormalise(predictions, data_mean, data_std)


test_nois = [np.random.normal(0, stds[i], size = (timesteps)) for i in range(num_species)]
test_noise = np.transpose(test_nois)
dat = solve_ivp(kinetic_model, t, test_ic, t_eval = t_eval, method = "RK45")
data = np.transpose(dat.y) + test_noise
test = np.clip(data, 0, 1e99)

fig, ax = plt.subplots()
ax.set_title('Test Experiment')
ax.set_ylabel('Concentrations $(M)$')
ax.set_xlabel('Time $(h)$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(alpha = 0.5)

for i in range(num_species):
    ax.scatter(time, test[:, i], marker = marker[i], \
        s = 8, label = species[i], color = color_1[i])
    ax.plot(time, unnormalised_predictions[:, i], \
        label = str(species[i] + ' predicted'), color = color_1[i])
    # ax.fill_between(
    #     time.squeeze(), unnormalised_predictions[:, i] - 2 * np.sqrt(variances[:, i]), \
    #     unnormalised_predictions[:, i] + 2 * np.sqrt(variances[:, i]), \
    #     alpha = 0.2, color = color_3[i], label = '95% CI'
    #     )
    # ax.plot(time, unnormalised_predictions[:, i] - 2 * np.sqrt(variances[:, i]), \
    #     color = color_3[i], linestyle = '--', linewidth = 1)
    # ax.plot(time, unnormalised_predictions[:, i] + 2 * np.sqrt(variances[:, i]), \
    #     color = color_3[i], linestyle = '--', linewidth = 1)
ax.legend()
file_path = 'Decomposition of Nitrous Oxide/Gaussian_process_prediction_test_noisy.png'
fig.savefig(file_path, dpi = 600)


"##############################################################################"
"###################### Model-Based Design of Experiments #####################"
"##############################################################################"

def GP_predict(test_ic, data_mean, data_std, species, timesteps, posterior,\
    learned_params, D, likelihood):
    norm_ic_test = (test_ic - data_mean) / data_std
    num_species = len(species)
    new_input = norm_ic_test.reshape(-1, num_species)
    predictions = np.empty((timesteps, num_species))
    variances = np.empty((timesteps, num_species))
    latent_dist = {}
    predictive_dist = {}
    predictive_mean = {}
    predictive_std = {}

    for i in range(timesteps):
        predictions[i] = new_input.reshape(num_species)

        for j in range(num_species):
            latent_dist[species[j]] = posterior[species[j]](
                learned_params[species[j]], D[species[j]])(new_input)
            predictive_dist[species[j]] = likelihood[species[j]](
                learned_params[species[j]], latent_dist[species[j]]
            )
            predictive_mean[species[j]] = predictive_dist[species[j]].mean()
            predictive_std[species[j]] = predictive_dist[species[j]].stddev()
            new_input[0][j] = predictive_mean[species[j]]

            if i == 0:
                variances[i][j] = 0
            
            else:
                variances[i][j] = predictive_std[species[j]]

        unnormalised_predictions = unnormalise(predictions, data_mean, data_std)
        unnormalised_variance = variances * (data_std**2)

    return unnormalised_predictions, unnormalised_variance

def SR_model(z0, equations, t, t_eval):
    i = 0

    for equation in equations:
        equation = str(equation)
        equation = equation.replace("CNO", "z[0]")
        equation = equation.replace("CN", "z[1]")
        equation = equation.replace("CO", "z[2]")
        equations[i] = equation
        i += 1

    def nest(t, z):
        dNOdt = eval(str(equations[0]))
        dNdt = (-1) * eval(str(equations[0]))
        dOdt = (-1/2) * eval(str(equations[0]))
        dzdt = [dNOdt, dNdt, dOdt]
        return dzdt

    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45")  

    return sol.y

def MBDoE(ic, data_mean, data_std, species, time, sym_model, posterior, \
    learned_params, D, likelihood):
    timesteps = len(time)
    SR_thing = SR_model(ic, sym_model, [0, np.max(time)], list(time))
    SR_thing = SR_thing.reshape(len(time), -1)
    GP_thing, _ = GP_predict(test_ic, data_mean, data_std, species, timesteps,
    posterior, learned_params, D, likelihood)
    difference = -np.sum((SR_thing - GP_thing)**2)
    return difference

def Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    data_mean, data_std, species, time, sym_model, posterior, learned_params, \
    D, likelihood):
    localsol = np.empty([multistart, number_parameters])
    localval = np.empty([multistart, 1])
    boundss = tuple([(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))])
    
    for i in range(multistart):
        x0 = np.random.uniform(lower_bound, upper_bound, size = number_parameters)
        res = minimize(to_opt, x0, args = (data_mean, data_std, species, time, \
                        sym_model, posterior, learned_params, D, likelihood), \
                        method = 'L-BFGS-B', bounds = boundss)
        localsol[i] = res.x
        localval[i] = res.fun

    minindex = np.argmin(localval)
    opt_val = localval[minindex]
    opt_param = localsol[minindex]
    
    return opt_val, opt_param


"##############################################################################"
"####################### MBDoE on GP model and SR model #######################"
"##############################################################################"

# from sym_reg_models import all_ODEs, best_model_index

multistart = 1
number_parameters = num_species
lower_bound = np.array([5 , 0, 0])
upper_bound = np.array([10, 2, 3])
to_opt = MBDoE
# sym_model = list(all_ODEs[best_model_index])

real_model = list((
    '(-2 * CNO**2) / (1 + 5 * CNO)',
    ))

a, b = Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    data_mean, data_std, species, time, real_model, posterior, learned_params, \
    D, likelihood)

print('Optimal experiment: ', b)


Title = "MBDoE GP Model vs Real Model"

STD = 0.0
noise = np.random.normal(0, STD, size = (number_parameters, timesteps))

y   = SR_model(b, real_model , [0, np.max(time)], list(time))
yy, _ = GP_predict(b, data_mean, data_std, species, timesteps, posterior, learned_params, D, likelihood)

fig, ax = plt.subplots()
ax.set_title(Title)
ax.set_ylabel("Concentration $(M)$")
ax.set_xlabel("Time $(h)$")

color_1 = cm.viridis(np.linspace(0, 1, number_parameters))
color_2 = cm.Wistia(np.linspace(0, 1, number_parameters))
color_3 = cm.cool(np.linspace(0, 1, number_parameters))

for j in range(number_parameters):
    ax.plot(time, np.clip(y[j] + noise[j], 0, 1e99), "x", markersize = 3, color = color_1[j])
    ax.plot(time, y[j], color = color_1[j], label = str('Real Model - ' + str(species[j])))
    ax.plot(time, yy[:, j], linestyle = 'dashed', color = color_1[j], label = str('GP Model - ' + str(species[j])))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(alpha = 0.5)
ax.legend()
file_path = 'Decomposition of Nitrous Oxide/MBDoE_real_and_GP_model.png'
fig.savefig(file_path, dpi = 600)

print(np.sum((y - yy.T)**2))