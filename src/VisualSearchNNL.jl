module VisualSearchNNL
    using CSV
    using Flux
    using Flux: logsumexp
    using Flux: params
    using VisualSearchACTR
    using ProgressMeter
    using Interpolations
    using KernelDensity
    using DataFrames
    using Distributions
    export make_training_batch
    export read_data
    export get_files
    export train_model

    include("flux_utilities.jl")
    include("functions.jl")
end
