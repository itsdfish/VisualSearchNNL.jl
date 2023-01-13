function rand_parms(;
        Δτ_dist = Uniform(.25,.75), 
        topdown_weight_dist = Uniform(0,2.5), 
        set_size_dist = 2:2:20
    )
    Δτ = rand(Δτ_dist)
    topdown_weight = rand(topdown_weight_dist)
    set_size = rand(set_size_dist)
    return (;Δτ,topdown_weight,set_size)
end

function make_training_batch(; n_parms, kwargs...)
    mapreduce(i -> begin 
                        mod(i, 1) == 0 ? println("parm set $i thread id $(Threads.threadid())") : nothing
                        make_training_data(; kwargs...)
                    end, 
            hcat, 1:n_parms)
end

"""
    make_training_data(; exp_parms, n_samples, n_trials, fixed_parms...)

Generates a set of training data where each column is a vector containing: rt, choice,
stimulus_present (2: present, 1: absent), set_size, Δτ, topdown_weight, LL

# Keywords 

- `exp_parms`: a NamedTuple of parameters for the experiment 
- `n_samples`: the number of samples for training data per parameter vector 
- `n_trials`: the number of trials used to estimate the kernel density 
- `fixed_parms...`: variable keyword arguments for model parameters 
"""
function make_training_data(; exp_parms, n_samples, n_trials, fixed_parms...)
    (;Δτ,topdown_weight,set_size) = rand_parms()
    data = zeros(Float32, 7, n_samples)
    # 1: absent, 2: present
    present = rand(1:2)
    kdes = create_kde(;n_trials, exp_parms, set_size, present, Δτ, topdown_weight, fixed_parms...)
    for c ∈ 1:n_samples       
        choice,rt = generate_data(1; set_size, exp_parms, present, Δτ, topdown_weight, fixed_parms...)
        LL = kde_logpdf(kdes, choice[1], rt[1])
        data[:,c] = [rt[1] choice[1] present set_size Δτ topdown_weight LL]
    end
    return data
end

generate_data(;present, set_size, exp_parms, parms...) = generate_data(1 ;present, set_size, exp_parms, parms...)

function generate_data(n_trials ;present, set_size, exp_parms, parms...)
    n_color_distractors = div(set_size, 2)
    n_shape_distractors = div(set_size, 2)
    experiment = Experiment(;n_trials, n_color_distractors, 
                            n_shape_distractors, set_size, exp_parms...)
    experiment.base_rate = present - 1
    run_condition!(experiment; parms...)
    responses = map(x -> x.response == :present ? 2 : 1, experiment.data)
    rts = map(x -> x.rt, experiment.data)
    return responses, rts
end

kde_logpdf(kdes, choice, rt) = log(max(10e-10, pdf(kdes[choice], rt)))

function create_kde(; n_trials, exp_parms, set_size, present, parms...)
    choices,rts = generate_data(n_trials; exp_parms, set_size, present, parms...)
    u_choices = 1:2
    probs = map(c -> mean(choices .== c), u_choices)
    return Dict(c => InterpKDE(robust_kernel(rts, choices, c, probs[c])) for c ∈ u_choices)
end

function robust_kernel(rts, choices, c, prob)
    # if empty, return a non-empty vector that will give a denisty of zero
    c_rts = rts[choices .==c]
    if isempty(c_rts)
        return kernel([eps(),2*eps()]) 
    end
    kd = kernel(c_rts)
    kd.density .*= prob
    return kd
end

function read_data(file_name; path=pwd())
    return Float32.(Array(CSV.read(path * "/" * file_name, DataFrame)))
end

function get_files(name_stem; path=pwd())
    all_files = readdir(path)
    files = filter(f -> contains(f, name_stem), all_files)
    return sort!(files)
end

kernel_dist(::Type{Epanechnikov}, w::Float64) = Epanechnikov(0.0, w)
kernel(data) = kde(data; kernel=Epanechnikov)