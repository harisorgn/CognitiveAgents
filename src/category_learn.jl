struct CLResult
    sol
    image_size
    σ_convolution
    subject_ID
    session
    run
end

probability_right_category(stim, w) = logistic(w' * stim)

probability_right_category(stim, w, β) = logistic(β * w' * stim)

function loglikelihood_category(P_right_cat, cat)
    return cat * log(P_right_cat + eps()) + (1 - cat) * log(1 - P_right_cat + eps())
end

function prediction_error(correct_cat, predicted_cat, η)
    return η * (correct_cat - predicted_cat)
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

        weights .+= stimulus .* prediction_error(correct_cat, P_right, η) 
    end

    return -llhood
end

function fit_CL(df, alg; σ_conv=5, grid_sz=(50,50), kwargs...)
    choices = get_choices(df)
    corrects = get_correct_categories(df)
    S = get_stimuli(df; grid_sz, σ_conv)

    lb = [0, 0]
    ub = [1.0, Inf]

    obj = OptimizationFunction(
        (p, hyperp) -> objective(S, choices, corrects, p), 
        Optimization.AutoForwardDiff()
    )
    p0 = [0.1, 1]
    prob = OptimizationProblem(obj, p0, lb = lb, ub = ub)
    sol = solve(prob, alg; kwargs...)

    id = unique(df.subject_id)
    session = unique(df.session)
    run = unique(df.run)

    res = CLResult(sol, grid_sz, σ_conv, id, session, run)

    return res
end

function run_CL_task(S::AbstractMatrix, corrects::AbstractVector, res::CLResult)
    η, β = res.sol
    D, N_trials = size(S)
    weights = zeros(typeof(η), D)

    choices = zeros(Int, N_trials)
    iscorrects = zeros(Bool, N_trials)
    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        correct_cat = corrects[t]

        P_right = probability_right_category(stimulus, weights, β)
        choices[t] = rand(Bernoulli(P_right))
        iscorrects[t] = correct_cat == choices[t]

        weights .+= η * (correct_cat - P_right) .* stimulus 
    end

    df = DataFrame(response = choices, correct = iscorrects)

    return df
end

function run_CL_task(df::DataFrame, res::CLResult; N_runs = 1)
    corrects = get_correct_categories(df)
    S = get_stimuli(df; grid_sz = res.image_size, σ_conv = res.σ_convolution)
   
    df_sim = mapreduce(append!, Base.OneTo(N_runs)) do _
        df_choices = run_CL_task(S, corrects, res)
        df_temp = copy(df)
        df_temp.response = df_choices.response
        df_temp.correct = df_choices.correct
        df_temp
    end
    
    return df_sim
end

function results_to_regressors(res::CLResult, df)
    df_regress = DataFrame(
        t = Float64[],
        t_RPE_pos = Float64[],
        t_RPE_neg = Float64[], 
        P_chosen = Float64[], 
        P_unchosen = Float64[], 
        P_left = Float64[], 
        P_right = Float64[], 
        RPE_pos = Float64[],
        RPE_neg = Float64[]
    )

    subject_ID = only(res.subject_ID)
    run = only(res.run)
    session = only(res.session)

    df_fit = @subset(df, :subject_id .== subject_ID, :run .== run, :session .== session)
    ST = parse.(Float64, df_fit.stim_presentation_time)
    RT = parse.(Float64, df_fit.response_time)

    choices = get_choicesp1(df_fit)
    corrects = get_corrects(df_fit)
    S = get_stimuli(df_fit; grid_sz=res.image_size, σ_conv=res.σ_convolution)
    D, N_trials = size(S)

    η, β = res.sol
    weights = zeros(typeof(η), D)
    for t in Base.OneTo(N_trials)
        stimulus = S[:,t]
        choice_cat = choices[t] 
        correct_cat = corrects[t]

        P_right = probability_right_category(stimulus, weights, β)
        Ps = [1 - P_right, P_right]
        idx_unchosen = choice_cat == 2 ? 1 : 2

        RPE = prediction_error(correct_cat, P_right, η)

        weights .+= η * (correct_cat - P_right) .* stimulus

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
                t = ST[t], 
                t_RPE_pos = t_RPE_pos,
                t_RPE_neg = t_RPE_neg, 
                P_chosen = Ps[choice_cat], 
                P_unchosen = Ps[idx_unchosen], 
                P_left = Ps[1], 
                P_right = Ps[2], 
                RPE_pos = RPE_pos,
                RPE_neg = RPE_neg
            )
        )
    end

    CSV.write("CL_regress_sub-$(subject_ID)_ses-$(session)_run-$(run).csv", df_regress)
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

