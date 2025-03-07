@model function category_match(response_dots, dot_evidence, choices)
    #σ ~ InverseGamma(2,8)
    β ~ LogNormal(4,1)
    P_lapse ~ Beta(1,20)

    N_trials = length(response_dots)
    for t in Base.OneTo(N_trials)
        RD = response_dots[t]
        evidence = dot_evidence[t]
        z_left, z_right = sum(evidence[1:RD, :]; dims=1)

        #β = pi / (σ * sqrt(6*(RD - 1)))
        P_left = logistic(β * (z_left - z_right))
        P_choices = [P_left, 1 - P_left]
        
        choice_idx = choices[t] + 1
        P_choice = P_choices[choice_idx] * (1 - P_lapse) + P_lapse / 2
        #P_choice = P_choices[choice_idx]
        choices[t] ~ Bernoulli(P_choice)
    end

    return (; β, P_lapse, choices)
    #return (; β, choices)
end

function probability_choices(Delta_loglikelihoods::Float64, β, P_lapse, )
    P_left = logistic(β * Delta_loglikelihoods)
    
    return [P_left, 1 - P_left] .* (1 - P_lapse) .+ P_lapse / 2
end

function probability_choices(loglikelihoods::AbstractMatrix, response_dots, β, P_lapse)
    z_left, z_right = sum(loglikelihoods[1:response_dots, :]; dims=1)
    Delta_loglikelihoods = z_left - z_right
    
    return probability_choices(Delta_loglikelihoods, β, P_lapse)
end

function objective(p, dot_evidence, choices, response_dots)
    β, P_lapse = p

    N_trials = length(dot_evidence)

    loglikelihood = 0.0

    for t in Base.OneTo(N_trials)
        loglikelihoods = dot_evidence[t]
        RD = response_dots[t]

        #β = pi / (σ * sqrt(6*(RD - 1)))
        
        P_choices = probability_choices(loglikelihoods, RD, β, P_lapse)
        choice_idx = choices[t] + 1
        P_choice = P_choices[choice_idx] 
       
        loglikelihood += log(P_choice)
    end

    return -loglikelihood
end

struct CMResult
    sol
    subject_ID
    session
    run
end

function fit_CM_bayes(df; kwargs...)
    L = get_loglikelihood_dots(df)
    C = get_choices(df)
    RD = get_response_dots(df)

    model = discrete_evidence(RD, L, C)
    chain = sample(model, NUTS(), 2_000, progress=false)

    return chain
end

function fit_CM_optimization(df, alg; kwargs...)
    L = get_loglikelihood_dots(df)
    C = get_choices(df)
    RD = get_response_dots(df)

    obj = OptimizationFunction(
        (p, hyperp) -> objective(p, L, C, RD),
        Optimization.AutoForwardDiff()
    )
    p0 = [2.0, 0.05]
    prob = OptimizationProblem(obj, p0, lb = [0.0, 0.0], ub = [100.0, 1.0])
    sol = solve(prob, alg; kwargs...)

    return CMResult(sol, unique(df.subject_id), unique(df.session), unique(df.run))
end

function results_to_regressors(res::CMResult, df; inter_dot_interval = 0.55)
    df_regress = DataFrame(t = Float64[], P_chosen = Float64[], P_unchosen = Float64[], P_left = Float64[], P_right = Float64[])

    df_fit = df[(df.subject_id .== res.subject_ID) .& (df.run .== res.run) .& (df.session .== res.session), :]

    ST = parse.(Float64, df_fit.stim_presentation_time)
    RT = parse.(Float64, df_fit.response_time)

    RD = get_response_dots(df_fit)
    L = get_loglikelihood_dots(df_fit)
    C = choicesp1(df_fit)

    β, σ_inf = res.sol
    for t in eachindex(RT)
        for d in Base.OneTo(RD[t])
            t_dot = ST[t] + d*inter_dot_interval
            P = probability_choices(L[t], d, σ_inf, β)
            idx_unchosen = C[t] == 1 ? 2 : 1
            push!(
                df_regress, 
                (t = t_dot, P_chosen = P[C[t]], P_unchosen = P[idx_unchosen], P_left = P[1], P_right = P[2])
            )
        end
    end

    CSV.write("CM_regress_sub-$(res.subject_ID)_ses-$(res.session)_run-$(res.run).csv", df_regress)
end

function results_to_dataframe(results::CMResult)
    df = DataFrame(
        subject_ID = Int64[],
        run = Int64[],
        session = String[],
        β = Float64[],
        σ_inf = Float64[]
    )

    for r in results

        push!(
            df,
            (
                subject_ID = r.subject_ID,
                run = r.run,
                session = r.session,
                β = r.sol[1],
                σ_inf = r.sol[2]
            ) 
        )
    end
    sort!(df, :subject_ID)

    return df
end