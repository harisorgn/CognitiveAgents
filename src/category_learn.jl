abstract type AbstractRegressor end

mutable struct EMAgent{T, R <: AbstractRegressor}
    W::Matrix{T}
    S̄::Vector{Matrix{T}}
    N::Vector{T}
    logpost::Vector{T}
    logpost_stim::Vector{T} # log P(z|stim) posterior without category info
    loglhood_stim::Vector{T}
    loglhood_category::Vector{T}
    logprior::Vector{T}
    RPE::Vector{T}
    z::Vector{Int}
    z_stim::Vector{Int}
    const N_loops::Int
    const α::T
    const β::T
    const η::T
    const ηₓ::T
    const σ²::T
    const s::T
    const regressor::R

    function EMAgent(X; N_loops=1, α=0.2, η=0.2, β=1, ηₓ=0.1, σ²=0.1, s=1.0)
        D, N_trials = size(X)
        logpost = zeros(typeof(s), 1)
        logpost_stim = zeros(typeof(s), 1)
        loglhood_stim = zeros(typeof(s), 1)
        loglhood_category = zeros(typeof(s), 1)
        logprior = zeros(typeof(s), 1)
        RPE = zeros(typeof(s), 1)
        N = zeros(typeof(s), 1)
        z = Int[]
        z_stim = Int[]
        W = zeros(typeof(s), D, 1)
        S̄ = [zeros(typeof(s), D, 2)]
        regressor = EMRegressor(N_trials, typeof(s))

        new{typeof(η), typeof(regressor)}(W, S̄, N, logpost, logpost_stim, loglhood_stim, loglhood_category, logprior, RPE, z, z_stim, N_loops, α, β, η, ηₓ, σ², s, regressor)
    end
end

struct CLResult
    sol
    image_size
    σ_convolution
    subject_ID
    session
    run
end

struct EMRegressor{T} <: AbstractRegressor
    RPE::Vector{T}
    loglikelihood_stimulus::Vector{T}
    loglikelihood_category::Vector{T}
    logposterior_stim::Vector{T}
    logposterior::Vector{T}

    function EMRegressor(N_trials::Int, T::DataType)
        new{T}(
            Vector{T}(undef, N_trials),
            Vector{T}(undef, N_trials), 
            Vector{T}(undef, N_trials), 
            Vector{T}(undef, N_trials),
            Vector{T}(undef, N_trials)
        )
    end
end

function add_regressors!(r::EMRegressor, agent::EMAgent, t)
    z = last(agent.z)
    z_stim = last(agent.z_stim)

    r.RPE[t] = agent.RPE[z_stim]
    r.loglikelihood_stimulus[t] = agent.loglhood_stim[z_stim]
    r.loglikelihood_category[t] = agent.loglhood_category[z]
    r.logposterior_stim[t] = agent.logpost_stim[z_stim]
    r.logposterior[t] = agent.logpost[z_stim]
end

struct NoRegressor <: AbstractRegressor end

function add_regressors!(::NoRegressor, ::EMAgent, t) end

probability_right_category(stim, w; β=1) = logistic(β * w' * stim)

function loglikelihood_category(P_right_cat, cat)
    return cat * log(P_right_cat + eps()) + (1 - cat) * log(1 - P_right_cat + eps())
end

function loglikelihood_stimulus(stim, S̄, N; σ²=0.1)
    μ₀ = -3
    σ₀² = 1

    #μ₀ = 0.1
    #σ₀² = 0.04
    
    D, N_cats = size(S̄)

    loglhood_cats = map(Base.OneTo(N_cats)) do c
        S̄_c = S̄[:,c]
        Ŝ = (S̄_c * N * σ₀² .+ μ₀ * σ²) / (N*σ₀² + σ²)
        #ν² = σ² + (σ² * σ₀²) / (N*σ₀² + σ²)
        ν² = (σ² * σ₀²) / (N*σ₀² + σ²)

        mean([-log(sqrt(ν²))-log(2*pi)/2-((stim[p] - Ŝ[p]).^2)/(2*ν²) for p in Base.OneTo(D)])
    end 
    
    return logsumexp(loglhood_cats)
end

kernel(t1, t2; L=1) = L / (t1 - t2)

function logprior(k, z, t; s=0)
    t_past = Base.OneTo(t-1)
    
    return log(sum(kernel.((t,), t_past[z .== k]))) + s
end

function update_latent_factor!(zs, logpost)
    zₜ = argmax(logpost)
    push!(zs, zₜ)
end

function local_MAP!(agent::EMAgent, x, correct_cat)    
    
    update_latent_factor!(agent.z, agent.logpost)
    update_latent_factor!(agent.z_stim, agent.logpost_stim)

    D, K = size(agent.W)
    zₜ = last(agent.z)

    if zₜ >= K
        push!(agent.N, zero(eltype(agent.N)))
        push!(agent.logpost, zero(eltype(agent.logpost)))
        push!(agent.logpost_stim, zero(eltype(agent.logpost_stim)))
        push!(agent.loglhood_stim, zero(eltype(agent.loglhood_stim)))
        push!(agent.loglhood_category, zero(eltype(agent.loglhood_category)))
        push!(agent.logprior, zero(eltype(agent.logprior)))
        push!(agent.RPE, zero(eltype(agent.RPE)))

        agent.W = hcat(agent.W, zeros(eltype(agent.W), D))
        push!(agent.S̄, zeros(eltype(first(agent.S̄)), D, 2))
    end

    agent.N[zₜ] += 1.0

    S̄_correct_cat = @views agent.S̄[zₜ][:, correct_cat + 1]
    S̄_correct_cat .+= agent.ηₓ * (x - S̄_correct_cat)
end

function E_step_stimulus!(agent::EMAgent, stim, t)
    @unpack W, S̄, N, z, logpost, α, β, σ², s = agent

    K = length(logpost)

    for k in Base.OneTo(K)
        agent.logprior[k] = k == K ? log(α) : logprior(k, z, t; s)
        S̄_k = @views S̄[k]
        loglhood_stim = loglikelihood_stimulus(stim, S̄_k, N[k]; σ²)
        agent.loglhood_stim[k] = loglhood_stim
        agent.logpost_stim[k] = loglhood_stim
    end
    
    agent.logprior .-= logsumexp(agent.logprior)

    agent.logpost_stim .+= agent.logprior
    agent.logpost_stim .-= logsumexp(agent.logpost_stim)
end

function E_step_category!(agent::EMAgent, stim, correct_category, t)
    @unpack W, S̄, N, z, logpost, α, β, σ², s = agent

    K = length(logpost)

    for k in Base.OneTo(K)
        P = probability_right_category(stim, W[:,k]; β)
        agent.loglhood_category[k] = loglikelihood_category(P, correct_category) 

        agent.logpost[k] = agent.loglhood_stim[k] + agent.loglhood_category[k]
    end
    
    agent.logpost .+= agent.logprior
    agent.logpost .-= logsumexp(logpost)
end

function prediction_error(correct_cat, predicted_cat, η, logpost)
    q = exp(logpost)

    return η * q * (correct_cat - predicted_cat)
end

function M_step!(agent::EMAgent, stim, correct_cat)
    for k in eachindex(agent.logpost)
        predicted_cat = probability_right_category(stim, agent.W[:,k]; β=agent.β)
        RPE = prediction_error(correct_cat, predicted_cat, agent.η, agent.logpost[k])
        agent.RPE[k] = RPE
        agent.W[:,k] .+= stim .* RPE 
    end
end

function loglikelihood(agent::EMAgent, stim::AbstractVector, choice_cat::Real)
    @unpack W, logpost_stim = agent

    K = length(logpost_stim)

    loglhood_category = map(Base.OneTo(K)) do k    
        P = probability_right_category(stim, W[:,k]; β=agent.β)
        loglikelihood_category(P, choice_cat) 
    end
    
    L_data = logsumexp(loglhood_category .+ logpost_stim) # log ∑ᶻ P(category|z) * P(z|stim)

    return L_data
end

function loglikelihood!(agent::EMAgent, S::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector; N_loops=1)
    D, N_trials = size(S)

    L_data = 0

    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        choice_cat = choices[t]
        correct_cat = corrects[t]

        loglikelihood_trial = 0
        for _ in Base.OneTo(N_loops)
            E_step_stimulus!(agent, stimulus, t)
            loglikelihood_trial = loglikelihood(agent, stimulus, choice_cat)
            E_step_category!(agent, stimulus, correct_cat, t)
            M_step!(agent, stimulus, correct_cat)
        end
        local_MAP!(agent, stimulus, correct_cat)

        L_data += loglikelihood_trial
        add_regressors!(agent.regressor, agent, t)
    end

    return L_data
end

initialise_agent(X; η=0.1, ηₓ=0.05, α=1.0, β=1.0, σ²=12, s=0.0) = EMAgent(X; η, ηₓ, α, β, σ², s)

function negative_loglikelihood(η, β, s, X::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector; ub_β = 100.0, ub_s = 100.0, N_loops=1)
    #=
    η = logistic(params[1])
    β = logistic(params[2]) * ub_β
    s = logistic(params[3]) * ub_s
    ag = initialise_agent(S; η=η, ηₓ=ηₓ, β=β, s=s)
    =#
    β = β * ub_β
    s = s * ub_s

    ag = initialise_agent(X; η, β, s)
    
    return -loglikelihood!(ag, X, choices, corrects; N_loops)
end


function fit_CL(df; σ_conv=5, grid_sz=(50,50), ub_β = 100.0, ub_s = 100.0, kwargs...)
    choices = get_choices(df)
    corrects = get_correct_categories(df)
    X = log.(get_stimuli(df; grid_sz, σ_conv)) # log-transform pixel values for greater resolution

    #=
    p0 = [0.1, 1.0, 1.0]
    lb = [0.0, 0.0, 0.0]
    ub = [1.0, 100.0, 100.0]
    
    obj = OptimizationFunction(
        (p, hyperp) -> negative_loglikelihood(p, S, choices, corrects),
        Optimization.AutoForwardDiff()
    )
   
    prob = OptimizationProblem(obj, p0, lb = lb, ub = ub)
    sol = solve(prob, alg; kwargs...)
    =#

    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.WARN, linear_solver=MumpsSolver))
    @variable(model, 0 <= η <= 1)
    @variable(model, 0 <= β <= 1)
    @variable(model, 0 <= s <= 1)
    @operator(model, neglhood, 3, (η, β, s) -> negative_loglikelihood(η, β, s, X, choices, corrects; ub_β, ub_s))
    @objective(model, Min, neglhood(η, β, s))

    optimize!(model)

    id = unique(df.subject_id)
    session = unique(df.session)
    run = unique(df.run)

    res = CLResult(model, grid_sz, σ_conv, id, session, run)

    return res
end

@model function category_learn(S::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector)
    η ~ Beta(2,4)
    β ~ InverseGamma(2,5)
    s ~ Exponential(4)

    Turing.@addlogprob! - negative_loglikelihood([η, β, s], S, choices, corrects)
end

function EM_learning!(agent::EMAgent, S::AbstractMatrix, corrects::AbstractVector; N_loops=1)
    D, N_trials = size(S)

    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        correct_cat = corrects[t]

        for _ in Base.OneTo(N_loops)
            E_step_stimulus!(agent, stimulus, t)

            E_step_category!(agent, stimulus, correct_cat, t)
    
            M_step!(agent, stimulus, correct_cat)
        end
        local_MAP!(agent, stimulus, correct_cat)
    end
end

function results_to_regressors(df_res, df_data; σ_conv=1, grid_sz=(50,50))
    for r in eachrow(df_res)
        df_regress = DataFrame(
            t_loglikelihood = Float64[],
            t_logposterior_stimulus = Float64[],
            t_logposterior = Float64[],
            t_RPE_pos = Float64[],
            t_RPE_neg = Float64[], 
            loglikelihood = Float64[],
            logposterior_stimulus = Float64[],
            logposterior = Float64[],
            RPE_pos = Float64[],
            RPE_neg = Float64[]
        )

        df_fit = df_data[(df_data.subject_id .== r.subject_id) .& (df_data.run .== r.run) .& (df_data.session .== r.session), :]
        ST = parse.(Float64, df_fit.stim_presentation_time)
        RT = parse.(Float64, df_fit.response_time)

        choices = get_choices(df_fit)
        corrects = get_correct_categories(df_fit)
        S = log.(get_stimuli(df_fit; grid_sz, σ_conv))
        D, N_trials = size(S)

        ag = initialise_agent(S; η=r.η, β=r.β, s=r.s)
        loglikelihood!(ag, S, choices, corrects)
        regressor = ag.regressor

        for t in Base.OneTo(N_trials)
            RPE = regressor.RPE[t]
            if RPE > 0.0
                t_RPE_pos = ST[t] + RT[t]
                RPE_pos = RPE
                t_RPE_neg = 0.0
                RPE_neg = 0.0
            else
                t_RPE_pos = 0.0
                RPE_pos = 0.0
                t_RPE_neg = ST[t] + RT[t]
                RPE_neg = RPE
            end

            push!(
                df_regress, 
                (
                    t_loglikelihood = ST[t],
                    t_logposterior_stimulus = ST[t],
                    t_logposterior = ST[t] + RT[t],
                    t_RPE_pos = t_RPE_pos,
                    t_RPE_neg = t_RPE_neg, 
                    loglikelihood = regressor.loglikelihood_stimulus[t],
                    logposterior_stimulus = regressor.logposterior_stim[t],
                    logposterior = regressor.logposterior[t],
                    RPE_pos = RPE_pos,
                    RPE_neg = RPE_neg,
                )
            )
        end

        CSV.write("CL_regress_sub-$(r.subject_id)_ses-$(r.session)_run-$(r.run).csv", df_regress)
    end
end

function results_to_dataframe(results::Vector{<:CLResult}; ub_β = 100.0, ub_s = 100.0)
    df = DataFrame(
        subject_id = Int64[],
        run = Int64[],
        session = String[],
        η = Float64[], 
        β = Float64[], 
        s = Float64[]
    )

    for r in results
        push!(
            df,
            (
                subject_id = only(r.subject_ID),
                run = only(r.run),
                session = only(r.session),
                η = value(variable_by_name(r.sol, "η")), 
                β = value(variable_by_name(r.sol, "β")) * ub_β, 
                s = value(variable_by_name(r.sol, "s")) * ub_s
            ) 
        )
    end
    sort!(df, :subject_id)

    return df
end

function get_categorization_rules(df_data::DataFrame, df_params::DataFrame, ID, session, run; σ_conv=5, grid_sz=(50,50))
    df_data_subj = @subset(df_data, :subject_id.==ID, :run.==run, :session.==session)
    df_params_subj = @subset(df_params, :subject_id.==ID, :run.==run, :session.==session)

    choices = get_choices(df_data_subj)
    corrects = get_correct_categories(df_data_subj)
    S = get_stimuli(df_data_subj; σ_conv, grid_sz)

    @assert nrow(df_params_subj)==1
    p = Vector(df_params_subj[1, [:η, :ηₓ, :α, :β, :σ², :s]])
    ag = initialise_agent(S; η=p[1], ηₓ=p[2], α=p[3], β=p[4], σ²=p[5], s=p[6])
    loglikelihood!(ag, S, choices, corrects)

    return ag.z
end


function act(agent::EMAgent, stimulus)
    z_stim_t = argmax(agent.logpost_stim)
    P = probability_right_category(stimulus, agent.W[:, z_stim_t]; β=agent.β)
    choice = Int(rand(Bernoulli(P)))

    return choice
end

function run_trial!(agent::EMAgent, env::CategoryLearnEnv, t)
    stimulus = observe(env)
    t = current_trial(env)

    E_step_stimulus!(agent, stimulus, t)
    
    choice = act(agent, stimulus)

    correct_category = act!(env, choice)

    M_step!(agent, stimulus, correct_category)

    E_step_category!(agent, stimulus, correct_category, t)

    local_MAP!(agent, stimulus, correct_category)
 
    increment!(env)

    return choice
end

function run_task!(agent, env)
    N_trials = env.N_trials

    choices = zeros(Int, N_trials)
    for t in Base.OneTo(N_trials)
        choices[t] = run_trial!(agent, env, t)
    end

    return choices
end

get_results(file) = deserialize(file)

#=
function E_step!(agent::EMAgent, stim, correct_category, t)
    @unpack W, S̄, N, z, logpost, α, β, σ², s = agent

    K = length(logpost)

    logpriors = similar(logpost)
    loglhood_category = zeros(K)

    for k in Base.OneTo(K)
        logpriors[k] = k == K ? log(α) : logprior(k, z, t; s)
        S̄_k = @views S̄[k]
        loglhood_stim = loglikelihood_stimulus(stim, S̄_k, N[k]; σ²)
        agent.loglhood_stim[k] = loglhood_stim

        P = probability_right_category(stim, W[:,k]; β)
        loglhood_category[k] = loglikelihood_category(P, correct_category) 

        agent.logpost[k] = loglhood_stim + loglhood_category[k]
        agent.logpost_stim[k] = loglhood_stim
    end
    
    logpriors .-= logsumexp(logpriors)   

    agent.logpost .+= logpriors 
    agent.logpost .-= logsumexp(logpost)

    agent.logpost_stim .+= logpriors
    agent.logpost_stim .-= logsumexp(agent.logpost_stim)
end
=#
