###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise, MKL, VisualSearchNNL, Plots, Flux, Distributions, Random, ProgressMeter
using ACTRModels, DataFrames, CSV, CUDA
using Flux: params
using BSON: @save
Random.seed!(5025)
###################################################################################################
#                                     load training data
###################################################################################################
path = "training_data/"
# training files 
training_files = get_files("training_data"; path)
# load the data files 
sim_data = mapreduce(f -> read_data(f; path), hcat, training_files)
# input values
train_x = sim_data[1:6,:]
# log likelihoods
train_y = sim_data[end,:]'
train_data = Flux.Data.DataLoader((train_x, train_y) |> gpu, batchsize=500)
###################################################################################################
#                                     load test data
###################################################################################################
# training files 
test_files = get_files("test_data"; path)
# load the data files 
sim_test_data = mapreduce(f -> read_data(f; path), hcat, test_files)
# inputs
test_x = sim_test_data[1:6,:]
# log likelihoods
test_y = sim_test_data[end,:]'
test_data = (x=test_x, y=test_y)
###################################################################################################
#                                        Create Network
###################################################################################################
# 6 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(6, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 120, tanh),
    Dense(120, 1, identity)) |> gpu

# check our model
params(model)

# loss function
loss_fn(a, b) = Flux.huber_loss(model(a), b) 

# optimization algorithm 
opt = ADAM(0.001)
###################################################################################################
#                                       Train Network
###################################################################################################
# number of Epochs to run
n_epochs = 50

# train the model
train_loss,test_loss = train_model(
    model, 
    n_epochs, 
    loss_fn, 
    train_data,
    train_x,
    train_y,
    test_data, 
    opt
)

# save the model for later
@save "visual_search.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# pyplot()
# plot the loss data
loss_plt = plot(1:n_epochs, train_loss, xlabel="Epochs", ylabel="Loss (huber)", label="training")
plot!(1:n_epochs, test_loss, label="test")

Random.seed!(8541)
idx = rand(1:size(test_x,2), 5000)
# predicted log likelihoods for test data
pred_y = model(test_x[:,idx])
sub_test_y = test_y[:,idx]
# residuals for test data
residual = pred_y .- sub_test_y

scatter(
    pred_y', 
    sub_test_y', 
    xlabel = "true LL", 
    ylabel = "predicted LL", 
    grid = false,
    leg = false,
    color = RGB(176/255, 73/255, 73/255),
    markerstrokewidth=1,
    markersize = 1.5,
    ylims = (-8,3),
    xlims = (-8,3),
    xaxis = font(7),
    yaxis = font(7),
    size = (240,130),
    dpi = 300,
)

i_x = -8:.01:4
plot!(i_x, i_x, color=:black, linestyle=:dash,
    linewidth=1)
savefig("scatter_LLs.png")

scatter(
    sub_test_y', 
    residual', 
    xlabel = "true LL", 
    ylabel = "residual", 
    grid = false,
    leg = false
)

# save LAN training fit statistics
ρ = cor(sub_test_y', pred_y')[1]
σ = std(residual, corrected=false)
df = DataFrame(ρ = [ρ], σ = [σ])
CSV.write("training_fit.csv", df)
