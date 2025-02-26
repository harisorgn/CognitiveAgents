@model function discrete_evidence(response_dots, dot_evidence, choices)
    #σ ~ InverseGamma(2,8)
    β ~ Uniform(0, 100)
    P_lapse ~ Beta(1, 5)

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
        choices[t] ~ Bernoulli(P_choice)
    end

    return (; β, P_lapse, choices)
end

function probability_choices(Lₜ, dots, σ, β)
    evidence = vec(sum(Lₜ[1:dots, :]; dims=1))
    samples = rand(MvNormal(evidence, dots*σ), 10_000)
    P = softmax(β*samples)

    return mean(P; dims=2)
end

function objective(p, L, choices, RD)
    β, σ_inf = p

    N_trials = length(L)

    loglikelihood = 0.0

    for t in Base.OneTo(N_trials)
        Lₜ = @views L[t]
        P = probability_choices(Lₜ, RD[t], σ_inf, β)
        loglikelihood += log(P[choices[t]])
    end

    return -loglikelihood
end

struct CMResult
    sol
    subject_ID
    session
    run
end

solution_to_params(sol) = (β = sol[1], σ_inf = sol[2])

function fit_bayes(df; kwargs...)
    L = get_loglikelihood_dots(df)
    C = get_choices(df)
    RD = get_response_dots(df)

    model = discrete_evidence(RD, L, C)
    chain = sample(model, NUTS(), 2_000, progress=false)

    return chain
end

function fit_discrete(df, alg; kwargs...)
    L = get_loglikelihood_dots(df)
    C = get_choicesp1(df)
    RD = get_response_dots(df)

    obj = OptimizationFunction(
        (p, hyperp) -> objective(p, L, C, RD)
    )
    p0 = [1.0, 0.5]
    prob = OptimizationProblem(obj, p0, lb = [0.0, 0.0], ub = [100.0, 100.0])
    sol = solve(prob, alg; kwargs...)

    return CMResult(sol, only(unique(df.subject_id)), only(unique(df.session)), only(unique(df.run)))
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