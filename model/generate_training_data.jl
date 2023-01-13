# for pid in $(pgrep "julia"); do cpulimit -l 90 -b -p $pid; done
###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using Revise, VisualSearchNNL, Random
using CSV, DataFrames, ThreadedIterables
###################################################################################################
#                                  generate training data
###################################################################################################
file_ids = [1:100;]
# number of parameter vectors for training per file
n_parms = 100
# number of test parms 
# number of sampled data points per parameter vector
n_samples = 200
# number of simulated trails for KDE 
n_trials = 25_000
ppi = 72
array_in = 5.95
letter_in = 0.45
# experiment parameters
exp_parms = (
        ppi,
        array_width = array_in * ppi,
        object_width = letter_in * ppi)

# fixed model parameters
parms = (
    σfactor = 1/3,
    viewing_distance = 30.0,
    bottomup_weight = 1.1)


make_training_batch(;n_parms=1, exp_parms, n_samples, n_trials, fixed_parms...)

tmap(i -> begin 
            Random.seed!(i)
            println("file id $i")
            sim_data = make_training_batch(;n_parms, exp_parms, n_samples, n_trials, fixed_parms...)
            # save training data 
            CSV.write(string("training_data/training_data", i ,".csv"), DataFrame(sim_data, :auto))
        end,
file_ids)
###################################################################################################
#                                     generate test data
###################################################################################################
test_file_ids = [1:10;]
tmap(i -> begin
            seed = 100_000 + i
            Random.seed!(i)
            println("file id $i")
            sim_data = make_training_batch(;n_parms, exp_parms, n_samples, n_trials, fixed_parms...)
            # save training data 
            CSV.write(string("training_data/test_data", i ,".csv"), DataFrame(sim_data, :auto))
        end,
test_file_ids)



# using VisualSearchNNL: generate_data
# using VisualSearchNNL: create_kde
# using VisualSearchNNL: kde_logpdf
# using VisualSearchNNL: make_training_data

# n_trials = 50_000
# n_samples = 100
# set_size = 10
# present = 2
# Δτ = .3
# top_down_weight = .66
# responses,rts =  generate_data(n_trials; present, set_size, exp_parms, fixed_parms...)

# kde_dict = create_kde(; n_trials, exp_parms, set_size, present, Δτ, top_down_weight, fixed_parms...)

# kde_logpdf(kde_dict, 2, .5)

# make_training_data(; exp_parms, n_samples, n_trials, fixed_parms...)