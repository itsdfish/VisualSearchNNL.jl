module VisualSearchNNL
    using Flux
    using VisualSearchACTR
    using ProgressMeter
    using Interpolations
    using KernelDensity
    export make_training_batch

    include("functions.jl")
    include("flux_utilities.jl")
end
