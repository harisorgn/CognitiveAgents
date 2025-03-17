
probability_right_category(stim, w) = logistic(w' * stim)

probability_right_category(stim, w, β) = logistic(β * w' * stim)

struct CLResult
    sol
    image_size
    σ_convolution
    subject_ID
    session
    run
end

function objective(S::AbstractMatrix, choices::AbstractVector, corrects::AbstractVector, p)
    #agent = initialise_agent(S, p) 
    #return -loglikelihood(agent, S, choices, corrects)

    η, β = p

    D, N_trials = size(S)
    weights = zeros(typeof(η), D)
    llhood = 0.0

    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        choice_cat = choices[t]
        correct_cat = corrects[t]

        P_right = probability_right_category(stimulus, weights, β)

        llhood += loglikelihood_category(P_right, choice_cat)

        weights .+= η * (correct_cat - P_right) .* stimulus 
    end

    return -llhood
end

function fit_CL(df, alg; σ_conv=5, grid_sz=(50,50), kwargs...)
    choices = get_choices(df)
    corrects = get_correct_categories(df)
    S = get_stimuli(df; grid_sz, σ_conv)

    lb = [0, 0]
    ub = [1.0, Inf]
    
    if alg isa GridSearch
        step_sz = alg.step_size
        r1 = lb[1]:step_sz:ub[1]
        r2 = lb[2]:step_sz:ub[2]
        OBJ = zeros(length(r1), length(r2))
        for (i, p1) in enumerate(r1)
            for (j, p2) in enumerate(r2)
                OBJ[i, j] = obj([p1, p2], nothing)
            end
        end

        sol = OBJ
    else
        obj = OptimizationFunction(
            (p, hyperp) -> objective(S, choices, corrects, p), 
            Optimization.AutoForwardDiff()
        )
        p0 = [0.1, 1]
        prob = OptimizationProblem(obj, p0, lb = lb, ub = ub)
        sol = solve(prob, alg; kwargs...)
    end

    id = unique(df.subject_id)
    session = unique(df.session)
    run = unique(df.run)

    res = CLResult(sol, grid_sz, σ_conv, id, session, run)

    return res
end

function run_CL_task(df::DataFrame, res::CLResult)
    corrects = get_correct_categories(df)
    S = get_stimuli(df; grid_sz = res.image_size, σ_conv = res.σ_convolution)
    η, β = res.sol

    D, N_trials = size(S)
    weights = zeros(typeof(η), D)

    choices = zeros(N_trials)
    iscorrects = zeros(Bool, N_trials)
    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        correct_cat = corrects[t]

        P_right = probability_right_category(stimulus, weights, β)
        choices[t] = rand(Bernoulli(P_right))
        iscorrects[t] = correct_cat == choices[t]

        weights .+= η * (correct_cat - P_right) .* stimulus 
    end

    df_sim = copy(df)
    df_sim.response .= choices
    df_sim.correct .= iscorrects

    return df_sim
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

function results_to_dataframe(results::CLResult)
    df = DataFrame(
        subject_ID = Int64[],
        run = Int64[],
        session = String[],
        η = Float64[],
        ηₓ = Float64[],
        α = Float64[]
    )

    for r in results

        push!(
            df,
            (
                subject_ID = r.subject_ID,
                run = r.run,
                session = r.session,
                η = r.sol[1],
                ηₓ = r.sol[2],
                α = r.sol[3]
            ) 
        )
    end
    sort!(df, :subject_ID)

    return df
end

