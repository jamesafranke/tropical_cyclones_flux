using MLDatasets
using Zarr
using NPZ
using Flux
using Flux: @functor, chunk, params
using Flux.Data: DataLoader
using Parameters: @with_kw
using BSON
using CUDA
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using Statistics
using DifferentialEquations
using Plots

function GaussianFourierProjection(embed_dim, scale)
    # Instantiate W once
    W = randn(Float32, embed_dim ÷ 2) .* scale
    # Return a function that always references the same W
    function GaussFourierProject(t)
        t_proj = t' .* W * Float32(2π)
        [sin.(t_proj); cos.(t_proj)]
    end
end

struct UNet
    layers::NamedTuple
end

marginal_prob_std(t, sigma=25.0f0) = sqrt.((sigma .^ (2t) .- 1.0f0) ./ 2.0f0 ./ log(sigma))

#User Facing API for UNet architecture.
function UNet(channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    return UNet((
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),
        # Encoding
        conv1=Conv((3, 3), 1 => channels[1], stride=1, bias=false),
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish),
        conv2=Conv((3, 3), channels[1] => channels[2], stride=2, bias=false),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish),
        conv3=Conv((3, 3), channels[2] => channels[3], stride=2, bias=false),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish),
        conv4=Conv((3, 3), channels[3] => channels[4], stride=2, bias=false),
        dense4=Dense(embed_dim, channels[4]),
        gnorm4=GroupNorm(channels[4], 32, swish),
        # Decoding
        tconv4=ConvTranspose((3, 3), channels[4] => channels[3], stride=2, bias=false),
        dense5=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish),
        tconv3=ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(0, -1, 0, -1), stride=2, bias=false),
        dense6=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish),
        tconv2=ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=(0, -1, 0, -1), stride=2, bias=false),
        dense7=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish),
        tconv1=ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride=1, bias=false),
    ))
end

@functor UNet

# helper to expand dims, similar to tensorflow expand dims
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

#the UNet struct callable and shows an example of a "Functional" API for modeling in Flux. \n
function (unet::UNet)(x, t)
    # Embedding
    embed = unet.layers.gaussfourierproj(t)
    embed = unet.layers.linear(embed)
    # Encoder
    h1 = unet.layers.conv1(x)
    h1 = h1 .+ expand_dims(unet.layers.dense1(embed), 2)
    h1 = unet.layers.gnorm1(h1)
    h2 = unet.layers.conv2(h1)
    h2 = h2 .+ expand_dims(unet.layers.dense2(embed), 2)
    h2 = unet.layers.gnorm2(h2)
    h3 = unet.layers.conv3(h2)
    h3 = h3 .+ expand_dims(unet.layers.dense3(embed), 2)
    h3 = unet.layers.gnorm3(h3)
    h4 = unet.layers.conv4(h3)
    h4 = h4 .+ expand_dims(unet.layers.dense4(embed), 2)
    h4 = unet.layers.gnorm4(h4)
    # Decoder
    h = unet.layers.tconv4(h4)
    h = h .+ expand_dims(unet.layers.dense5(embed), 2)
    h = unet.layers.tgnorm4(h)
    h = unet.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(unet.layers.dense6(embed), 2)
    h = unet.layers.tgnorm3(h)
    h = unet.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(unet.layers.dense7(embed), 2)
    h = unet.layers.tgnorm2(h)
    h = unet.layers.tconv1(cat(h, h1, dims=3))
    # Scaling Factor
    h ./ expand_dims(marginal_prob_std(t), 3)
end

function model_loss(model, x, ϵ=1.0f-5)
    batch_size = size(x)[end]
    # (batch) of random times to approximate 𝔼[⋅] wrt. 𝘪 ∼ 𝒰(0, 𝘛)
    random_t = rand!(similar(x, batch_size)) .* (1.0f0 - ϵ) .+ ϵ
    # (batch) of perturbations to approximate 𝔼[⋅] wrt. 𝘹(0) ∼ 𝒫₀(𝘹)
    z = randn!(similar(x))
    std = expand_dims(marginal_prob_std(random_t), 3)
    # (batch) of perturbed 𝘹(𝘵)'s to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    perturbed_x = x + z .* std
    # 𝘚₀(𝘹(𝘵), 𝘵)
    score = model(perturbed_x, random_t)
    # mean over batches
    mean(
        # L₂ norm over WHC dimensions
        sum((score .* std + z) .^ 2; dims=1:(ndims(x) - 1))
    )
end

#Helper function from DrWatson.jl to convert a struct to a dict
function struct2dict(::Type{DT}, s) where {DT<:AbstractDict}
    DT(x => getfield(s, x) for x in fieldnames(typeof(s)))
end
struct2dict(s) = struct2dict(Dict, s)

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-4                                        # learning rate
    batch_size = 32                                 # batch size
    epochs = 50                                     # number of epochs
    seed = 1                                        # random seed
    cuda = false                                    # use CPU
    verbose_freq = 10                               # logging for every verbose_freq iterations
    tblogger = true                                 # log training with tensorboard
    save_path = "output"                            # results path
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    xtrain = npzread("data/raw/mnist.npy")
    loader = DataLoader((xtrain), batchsize=32, shuffle=true)
    # initialize UNet model
    unet = UNet() |> device
    # ADAM optimizer
    opt = ADAM(args.η)
    # parameters
    ps = Flux.params(unet)
    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # Training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for x in loader
            x = device(x)
            loss, grad = Flux.withgradient(ps) do
                model_loss(unet, x)
            end
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss = loss
                end
            end
            train_steps += 1
        end
    end

    # save model
    model_path = joinpath(args.save_path, "model.bson")
    let unet = cpu(unet), args = struct2dict(args)
        BSON.@save model_path unet args
        @info "Model saved: $(model_path)"
    end
end

#if abspath(PROGRAM_FILE) == @__FILE__
    #train()
#end

train()


### Plotting ####

#Helper function yielding the diffusion coefficient from a SDE.
diffusion_coeff(t, sigma=convert(eltype(t), 25.0f0)) = sigma .^ t

#Helper function that produces images from a batch of images.
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# Helper to make an animation from a batch of images.
function convert_to_animation(x)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], batches),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], batches),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

#Helper function that generates inputs to a sampler.
function setup_sampler(device, num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_x = (
        randn(Float32, (28, 28, 1, num_images)) .*
        expand_dims(marginal_prob_std(t), 3)
    ) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

function DifferentialEquations_problem(model, init_x, time_steps, Δt)
    function f(u, p, t)
        batch_time_step = fill!(similar(u, size(u)[end]), 1) .* t
        return (
            -expand_dims(diffusion_coeff(batch_time_step), 3) .^ 2 .*
            model(u, batch_time_step)
        )
    end

    function g(u, p, t)
        batch_time_step = fill!(similar(u), 1) .* t
        diffusion_coeff(batch_time_step)
    end
    tspan = (time_steps[begin], time_steps[end])
    SDEProblem(f, g, init_x, tspan), ODEProblem(f, init_x, tspan)
end

function plot_result(unet, args)
    args = Args(; args...)
    args.seed > 0 && Random.seed!(args.seed)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    unet = unet |> device
    time_steps, Δt, init_x = setup_sampler(device)
    
    # Setup an SDEProblem and ODEProblem to input to `solve()`.
    # Use dt=Δt to make the sample paths comparable to calculating "by hand".
    sde_problem, ode_problem = DifferentialEquations_problem(unet, init_x, time_steps, Δt)

    @info "Probability Flow ODE Sampling w/ DifferentialEquations.jl"
    diff_eq_ode = solve(ode_problem, dt=Δt, adaptive=false)
    diff_eq_ode_end = diff_eq_ode[:, :, :, :, end]
    diff_eq_ode_images = convert_to_image(diff_eq_ode_end, size(diff_eq_ode_end)[end])
    save(joinpath(args.save_path, "diff_eq_ode_images.jpeg"), diff_eq_ode_images)
    diff_eq_ode_animation = convert_to_animation(diff_eq_ode)
    gif(diff_eq_ode_animation, joinpath(args.save_path, "diff_eq_ode.gif"), fps=50)
    ode_plot = plot(diff_eq_ode, title="Probability Flow ODE", legend=false, ylabel="x", la=0.25)
    plot!(time_steps, diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    plot!(time_steps, -diffusion_coeff(time_steps), xflip=true, ls=:dash, lc=:red)
    savefig(ode_plot, joinpath(args.save_path, "diff_eq_ode_plot.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load "output/model.bson" unet args
    #
    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    plot_result(unet, args)
end

