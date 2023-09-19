import Pkg
project_dir = "/rds/general/user/md1621/home/SymbolicRegression.jl"
Pkg.activate(project_dir)
Pkg.instantiate()

using IterTools: ncycle
using OrdinaryDiffEq
using SymbolicRegression
using Infiltrator
using DelimitedFiles

num_datasets = 7
num_states = 2

tspan = (0e0,1e1)
num_timepoints = 30

times_per_dataset=collect(range(tspan[begin], tspan[end]; length=num_timepoints))

ini_A = [2e0, 1e1, 2e0, 1e1, 1e1, 73194943e-7, 580158252e-8]
ini_B = [0e0, 0e0, 2e0, 2e0, 1e0, 2e0, 2e0]

initial_conditions = [[x0...] for x0 in zip(ini_A, ini_B)]

scoeff = [-1e0, 1e0]

function rate(A, B; k1 = 7, k2 = 3, k3 = 4, k4 = 2, k5 = 6)
    num = ((k1 * A) - (k2 * B))
    den = ((k3 * A) + (k4 * B) + k5)
    return num / den
end

function f(u, p, t)
    A, B = u
    r = rate(A, B)
    return [stoic * r for stoic in scoeff]
end

condition(u,t,integrator) = true #any(y -> y < 1f-2, u)
function affect!(integrator)
    @show integrator
    try
        step!(deepcopy(integrator))
    catch e
        @infiltrate
        throw(e)
    end
    return
end
dcb = DiscreteCallback(condition, affect!)

function terminate_affect!(integrator)
    terminate!(integrator)
end
function terminate_condition(u,t,integrator)
    any(y -> y < 0, u)
end
ccb = ContinuousCallback(terminate_condition,terminate_affect!)

isoutofdomain(u,p,t) = any(y -> y < 0, u)

function generate_datasets(; noise_per_concentration=nothing)
    datasets = []
    for ini in initial_conditions
        prob = ODEProblem(f, ini, tspan)
        sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat=times_per_dataset, callback=dcb, isoutofdomain)
        arr = Array(sol)
        if isnothing(noise_per_concentration)
            push!(datasets, Array(sol))
        else
            noise_matrix = vcat([noise_level * randn(Float32, (1,length(times_per_dataset))) for noise_level in noise_per_concentration]...)
            push!(datasets, Array(sol) .+ noise_matrix)
            # push!(datasets, Array(sol))
        end
    end
    return datasets
end


#------------------------------#
# GENERATE THE DATASET IN JULIA

# datasets_ = generate_datasets(; noise_per_concentration=[0.46878853f0, 0.5313656f0, 0.23121147f0])
# datasets_ = generate_datasets()

# READ THE DATASET FROM PYTHON
num_initial_conditions = num_datasets
datasets = [permutedims(readdlm(project_dir*"/data_SI_$i.csv", '|', Float64, '\n')) for i in 0:num_initial_conditions-1]
println(permutedims(readdlm(project_dir*"/data_SI_$i.csv", '|', Float64, '\n')) for i in 0:num_initial_conditions-1)
#------------------------------#


X = hcat(datasets...)
times = ncycle(times_per_dataset, num_datasets) |> collect
experiments = vcat([fill(Float64(i), num_timepoints) for i in 1:num_datasets]...)

y = [1e0, 2e0]

options = SymbolicRegression.Options(binary_operators=(+, *, /, -),hofFile="hall_of_fame_SI.csv",maxsize=25,parsimony=1e-5)
hall_of_fame = EquationSearch(X, y, niterations=20, options=options, numprocs=8, times=times, experiments=experiments, stoic_coeff=scoeff)

dominating = calculate_pareto_frontier(X, y, hall_of_fame, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member.tree, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end

#=
# test against symbolic solution
proposed_rate(x1,x2) = ((x1 - ((x2 - x1) / 1.3333641f0)) / ((((x2 - -0.15033427f0) * 1.500032f0) - x2) + (x1 + 1.2743267f0)))
function f_(u,p,t)
    Ca, Cb = u
    r = proposed_rate(Ca, Cb)
    return [stoic * r for stoic in scoeff]
end
datasets_ = []
for ini in initial_conditions
    prob = ODEProblem(f_, ini, tspan)
    sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat=times_per_dataset)
    push!(datasets_, Array(sol))
end
=#
