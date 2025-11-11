mutable struct EMAgent
    W::Matrix{Float64}
    S̄::Vector{Matrix{Float64}}
    N::Vector{Int}
    logq::Vector{Float64}
    logq_stim::Vector{Float64} # log P(z|stim) posterior without category info
    loglhood_stim::Vector{Float64}
    z::Vector{Int}
    z_stim::Vector{Int}
    const N_loops::Int
    const L_data::Vector{Float64}
    const α::Float64
    const β::Float64
    const η::Float64
    const ηₓ::Float64
    const σ²::Float64

    function EMAgent(X; N_loops=1, α=0.2, η=0.2, β=1, ηₓ=0.1, σ²=0.1)
        D, N_trials = size(X)
        logq = zeros(1)
        logq_stim = zeros(1)
        loglhood_stim = zeros(1)
        N = [0]
        z = Int[]
        z_stim = Int[]
        W = zeros(D, 1)
        S̄ = [zeros(D, 2)]
        loglikelihood_data = zeros(N_trials)

        new(W, S̄, N, logq, logq_stim, loglhood_stim, z, z_stim, N_loops, loglikelihood_data, α, β, η, ηₓ, σ²)
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

struct TrialMetrics
    W::Vector{Matrix{Float64}}
    logq::Vector{Vector{Float64}}
    loglhood::Vector{Vector{Float64}}

    function TrialMetrics(N_trials)
        new(
            Vector{Matrix{Float64}}(undef, N_trials), 
            Vector{Vector{Float64}}(undef, N_trials), 
            Vector{Vector{Float64}}(undef, N_trials)
        )
    end
end

struct EmptyMetrics end

function store_trial_metrics!!(tm::TrialMetrics, t, agent)
    tm.W[t] = copy(agent.W)
    tm.logq[t] = copy(agent.logq)
    tm.loglhood[t] =  copy(agent.loglhood_stim)
end

function store_trial_metrics!!(tm::EmptyMetrics, t, agent) end

probability_right_category(stim, w) = logistic(w' * stim)

probability_right_category(stim, w, β) = logistic(β * w' * stim)

function loglikelihood_category(P_right_cat, cat)
    return cat * log(P_right_cat + eps()) + (1 - cat) * log(1 - P_right_cat + eps())
end

function loglikelihood_stimulus(stim, S̄, N; σ²=0.1)
    #μ₀ = -2
    #σ₀² = 1

    μ₀ = 0.1
    σ₀² = 0.04
    
    D, N_cats = size(S̄)

    loglhood_cats = map(Base.OneTo(N_cats)) do c
        S̄_c = S̄[:,c]
        Ŝ = (S̄_c * N * σ₀² .+ μ₀ * σ²) / (N*σ₀² + σ²)
        #ν² = σ² + (σ² * σ₀²) / (N*σ₀² + σ²)
        ν² = (σ² * σ₀²) / (N*σ₀² + σ²)
        
        #- D*log(2*pi*ν²)/2 - sum(log.(stim)) - sum((log.(stim) .- ŝ).^2 / (2*ν²))
        - D*log(2*pi*ν²)/2 - sum((stim .- Ŝ).^2 / (2*ν²))
    end 

    return logsumexp(loglhood_cats)
end

function logprior(k, z, t; β=1)
    t_past = Base.OneTo(t-1)
    
    return log(sum(kernel.(Ref(t), t_past[z .== k]; β)))
end

function update_latent_factor!(zs, logq)
    zₜ = argmax(logq)
    push!(zs, zₜ)
end

function local_MAP!(agent::EMAgent, x, correct_cat)    
    
    update_latent_factor!(agent.z, agent.logq)
    update_latent_factor!(agent.z_stim, agent.logq_stim)

    D, K = size(agent.W)
    zₜ = last(agent.z)

    if zₜ >= K
        push!(agent.N, 0)
        push!(agent.logq, 0)
        push!(agent.logq_stim, 0)
        push!(agent.loglhood_stim, 0)

        agent.W = hcat(agent.W, zeros(D))
        push!(agent.S̄, zeros(D, 2))
    end

    agent.N[zₜ] += 1

    S̄_correct_cat = @views agent.S̄[zₜ][:, correct_cat + 1]
    #S̄_correct_cat .+= agent.ηₓ * (log.(x) - S̄_correct_cat)
    S̄_correct_cat .+= agent.ηₓ * (x - S̄_correct_cat)
end

function E_step!(agent::EMAgent, stim, correct_category, t)
    @unpack W, S̄, N, z, logq, α, β, σ² = agent

    K = length(logq)

    logpriors = similar(logq)
    loglhood_category = zeros(K)

    for k in Base.OneTo(K)
        logpriors[k] = k == K ? log(α) : logprior(k, z, t; β)
        S̄_k = @views S̄[k]
        loglhood_stim = loglikelihood_stimulus(stim, S̄_k, N[k]; σ²)
        agent.loglhood_stim[k] = loglhood_stim

        P = probability_right_category(stim, W[:,k])
        loglhood_category[k] = loglikelihood_category(P, correct_category) 

        agent.logq[k] = loglhood_stim + loglhood_category[k]
        agent.logq_stim[k] = loglhood_stim
    end
    
    logpriors .-= logsumexp(logpriors)   

    agent.logq .+= logpriors 
    agent.logq .-= logsumexp(logq)

    agent.logq_stim .+= logpriors
    agent.logq_stim .-= logsumexp(agent.logq_stim)
end

function E_step_stimulus!(agent::EMAgent, stim, t)
    @unpack W, S̄, N, z, logq, α, β, σ² = agent

    K = length(logq)

    logpriors = similar(logq)

    for k in Base.OneTo(K)
        logpriors[k] = k == K ? log(α) : logprior(k, z, t; β)
        S̄_k = @views S̄[k]
        loglhood_stim = loglikelihood_stimulus(stim, S̄_k, N[k]; σ²)
        agent.loglhood_stim[k] = loglhood_stim
        agent.logq_stim[k] = loglhood_stim
    end
    
    logpriors .-= logsumexp(logpriors)   

    agent.logq_stim .+= logpriors
    agent.logq_stim .-= logsumexp(agent.logq_stim)
end

function E_step_category!(agent::EMAgent, stim, correct_category, t)
    @unpack W, S̄, N, z, logq, α, β, σ² = agent

    K = length(logq)

    logpriors = similar(logq)
    loglhood_category = zeros(K)

    for k in Base.OneTo(K)
        logpriors[k] = k == K ? log(α) : logprior(k, z, t; β)
        S̄_k = @views S̄[k]
        loglhood_stim = loglikelihood_stimulus(stim, S̄_k, N[k]; σ²)
        agent.loglhood_stim[k] = loglhood_stim

        P = probability_right_category(stim, W[:,k])
        loglhood_category[k] = loglikelihood_category(P, correct_category) 

        agent.logq[k] = loglhood_stim + loglhood_category[k]
    end
    
    logpriors .-= logsumexp(logpriors)   

    agent.logq .+= logpriors 
    agent.logq .-= logsumexp(logq)
end

function prediction_error(correct_cat, predicted_cat, η, logq)
    q = exp(logq)

    return η * q * (correct_cat - predicted_cat)
end

function M_step!(agent::EMAgent, stim, correct_cat)
    for k in eachindex(agent.logq)
        predicted_cat = probability_right_category(stim, agent.W[:,k])
        agent.W[:,k] .+= stim .* prediction_error(correct_cat, predicted_cat, agent.η, agent.logq[k])
    end
end

function loglikelihood(agent::EMAgent, stim::AbstractVector, choice_cat::Real)
    @unpack W, logq_stim, logq = agent

    K = length(logq)

    loglhood_category = map(Base.OneTo(K)) do k    
        P = probability_right_category(stim, W[:,k])
        loglikelihood_category(P, choice_cat) 
    end
    
    L_data = logsumexp(loglhood_category .+ logq_stim) # log ∑ᶻ P(category|z) * P(z|stim)

    return L_data
end

function loglikelihood(agent::EMAgent, S::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector; N_loops=1)
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
    end

    return L_data
end

function negative_loglikelihood(params::AbstractVector, S::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector; N_loops=1)
    ag = initialise_agent(S, params)

    return -loglikelihood(ag, S, choices, corrects; N_loops)
end

function act(agent::EMAgent, stimulus)
    z_stim_t = argmax(agent.logq_stim)
    P = probability_right_category(stimulus, agent.W[:, z_stim_t])
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

initialise_agent(X, p) = EMAgent(X; η=p[1], ηₓ=p[2], α=p[3], β=p[4], σ²=p[5])
#initialise_agent(X, p) = EMAgent(X; η=0.1, ηₓ=0.1, α=p[1], β=p[2], σ²=0.1)

struct GridSearch
    step_size::Float64
end

function fit_EM(df, alg; σ_conv=5, grid_sz=(50,50), kwargs...)
    choices = get_choices(df)
    corrects = get_correct_categories(df)
    S = get_stimuli(df; grid_sz, σ_conv)

    p0 = [0.1, 0.1, 1.0, 1.0, 0.1]
    lb = [0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [1.0, 1.0, 100.0, 100.0, 100.0]
    
    obj = OptimizationFunction(
        (p, hyperp) -> negative_loglikelihood(p, S, choices, corrects)
    )
    prob = OptimizationProblem(obj, p0, lb = lb, ub = ub)
    sol = solve(prob, alg; kwargs...)

    id = unique(df.subject_id)
    session = unique(df.session)
    run = unique(df.run)

    res = CLResult(sol, grid_sz, σ_conv, id, session, run)

    return res
end

function results_to_regressors(res::CLResult, df)
    df_regress = DataFrame(
        t_loglikelihood = Float64[],
        t_RPE_pos = Float64[],
        t_RPE_neg = Float64[], 
        loglikelihood = Float64[], 
        RPE_pos = Float64[],
        RPE_neg = Float64[]
    )

    df_fit = df[(df.subject_id .== res.subject_ID) .& (df.run .== res.run) .& (df.session .== res.session), :]
    ST = parse.(Float64, df_fit.stim_presentation_time)
    RT = parse.(Float64, df_fit.response_time)

    C = choices(df_fit)
    X = stimuli(df_fit; grid_sz=res.image_size, σ_conv=res.σ_convolution)
    D, N_trials = size(X)

    ag = initialise_agent(X, res.sol)
    tm = TrialMetrics(N_trials)
    EM_learning!(ag, X, C, tm)

    for t in Base.OneTo(N_trials)
        
        z_stim = ag.z_stim[t]
        loglikelihood = tm.loglhood[t][z_stim] 

        z = ag.z[t]
        w = t == 1 ? zeros(D) : tm.W[t-1][:, z]
        logq = tm.logq[t][z]
        pred_cat = probability_right_category(X[:, t], w)
        RPE = prediction_error(C[t], pred_cat, ag.η, logq)

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
                t_RPE_pos = t_RPE_pos,
                t_RPE_neg = t_RPE_neg, 
                loglikelihood = loglikelihood, 
                RPE_pos = RPE_pos,
                RPE_neg = RPE_neg
            )
        )
    end

    CSV.write("CL_regress_sub-$(res.subject_ID)_ses-$(res.session)_run-$(res.run).csv", df_regress)
end

function results_to_dataframe(results::Vector{<:CLResult})
    df = DataFrame(
        subject_id = Int64[],
        run = Int64[],
        session = String[],
        η = Float64[], 
        ηₓ = Float64[], 
        α = Float64[], 
        β = Float64[], 
        σ² = Float64[]
    )

    for r in results
        push!(
            df,
            (
                subject_id = only(r.subject_ID),
                run = only(r.run),
                session = only(r.session),
                η = r.sol[1], 
                ηₓ = r.sol[2], 
                α = r.sol[3], 
                β = r.sol[4], 
                σ² = r.sol[5]
            ) 
        )
    end
    sort!(df, :subject_id)

    return df
end
