"""
    train_model(
        model, 
        n_epochs, 
        loss_fn, 
        train_data, 
        train_x, 
        train_y, 
        test_data, 
        opt; 
        show_progress = true
    )

Trains a neural network `model` given training data and test data. Returns a vector for training loss and test loss.
# Arguments 

- `model`: neural network model 
- `n_epochs`: number of epochs 
- `loss_fn`: loss function 
- `train_data`: batched training data
- `train_x`: training predictors 
- `train_y`: training criterion 
- `test_data`: NamedTuple of test data 
- `opt`: optimization algorithm 
# Keywords

- `show_progress`: show progress meter if true 
"""
function train_model(
    model, 
    n_epochs, 
    loss_fn, 
    train_data, 
    train_x, 
    train_y, 
    test_data, 
    opt; 
    show_progress = true
    )

    meter = Progress(n_epochs; enabled=show_progress)
    train_loss = zeros(n_epochs)
    test_loss = zeros(n_epochs)
    max_loss = -Inf
    min_loss = Inf
    @showprogress for i in 1:n_epochs
        Flux.train!(loss_fn, params(model), train_data, opt)
        train_loss[i] = loss_fn(train_x, train_y)
        test_loss[i] = loss_fn(test_data.x, test_data.y)
        loss = round(train_loss[i], digits=4)
        max_loss = loss >  max_loss ? loss : max_loss 
        min_loss = loss <  min_loss ? loss : min_loss
        values = [(:iter,i),(:loss,loss), (:max_loss, max_loss), (:min_loss,min_loss)]
        next!(meter; showvalues = values)
    end
    return train_loss,test_loss
end