# for pid in $(pgrep "julia"); do cpulimit -l 95 -b -p $pid; done
###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using Revise, VisualSearchNNL, Random, Flux
using CSV, DataFrames, Plots
using VisualSearchACTR
using VisualSearchNNL: generate_data
using BSON: @load
import_gui() # uncomment if using gui
###################################################################################################
#                                  generate training data
###################################################################################################
@load "visual_search.bson" model 
# number of simulated trails for KDE 
n_trials = 10_000
ppi = 72
array_in = 5.95
letter_in = 0.45
# experiment parameters
exp_parms = (
        visible = false,
        trace = false,
        ppi,
        array_width = array_in * ppi,
        object_width = letter_in * ppi)

# fixed model parameters
parms = (
    σfactor = 1/3,
    viewing_distance = 30.0,
    bottomup_weight = 1.1)

choice = 2
present = 2
set_size = 2
Δτ = .25
topdown_weight = .66

choices,all_rts = generate_data(n_trials; 
                                present, set_size, exp_parms, Δτ, topdown_weight, parms...)
p_choice = mean(choices .== choice)
rts = all_rts[choices .== choice]
# rt[1] choice[1] present set_size Δτ top_down_weight

ts = range(.2, 1.5, length=100)
LLs = map(t -> model([t, choice, present, set_size, Δτ, topdown_weight])[1], ts)
dens = exp.(LLs)
hist = histogram(rts, normalize=true)
hist[1][1][:y] .*= p_choice
plot!(ts, dens, grid=false, leg=false)
