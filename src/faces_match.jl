@model function ddm(data, conditions, N_conditions; min_rt=0.2)

    α ~ LogNormal(1, 0.5)
    τ ~ Uniform(0.0, min_rt)
    z ~ Beta(1.5, 1.5)

    drift_intercept ~ Normal(0, 1)
    drift_slope ~ Normal(0, 1)

    drift = drift_intercept .+ drift_slope .* conditions

    Turing.@addlogprob! sum(logpdf.(DDM.(drift, α, z, τ), data))

end

function objective(p, data, aggressiveness)
    α, τ, z, drift_intercept, drift_slope = p
    drifts = drift_intercept .+ (drift_slope .* aggressiveness)

    return -sum(logpdf.(DDM.(drifts, α, z, τ), data))
end

function add_data!(df, df_aggressive)
    df.choice = get_choicesp1(df)
    df.rt = get_response_times(df)
    df.score = df_aggressive.score
end

struct FacesResult
    sol
    subject_ID
    session
    run
end

function fit_faces(df, alg; min_rt = 0.2, kwargs...)
    df_agr = read_aggressiveness(df; normalize=true)
    add_data!(df, df_agr)

    filter!(:rt => rt -> !ismissing(rt) && rt >= min_rt, df)

    data = map(eachrow(df)) do r
        (choice = r.choice, rt = r.rt)
    end

    obj = OptimizationFunction(
        (p, hyperp) -> objective(p, data, df.score),
        Optimization.AutoForwardDiff()
    )
    p0 = [1.0, 0.1, 0.5, 0.0, 1.0]
    prob = OptimizationProblem(obj, p0, lb = [0.5, 0.0, 0.0, -Inf, -Inf], ub = [Inf, Inf, 1.0, Inf, Inf])
    sol = solve(prob, alg; kwargs...)

    id = unique(df.subject_id)
    session = unique(df.session)
    run = unique(df.run)

    res = FacesResult(sol, id, session, run)

    return res
end


function results_to_dataframe(results::Vector{<:FacesResult})
    df = DataFrame(
        subject_id = Int64[],
        run = Int64[],
        session = String[],
        α = Float64[],
        τ = Float64[],
        z = Float64[],
        drift_intercept = Float64[],
        drift_slope = Float64[],
        drift_angry = Float64[],
        drift_neutral = Float64[]
    )

    for r in results
        α, τ, z, drift_intercept, drift_slope = r.sol

        drift_angry = drift_intercept - 4.5 * drift_slope
        drift_neutral = drift_intercept + 4.5 * drift_slope

        push!(
            df,
            (
                subject_id = only(r.subject_ID),
                run = only(r.run),
                session = only(r.session),
                α = α,
                τ = τ,
                z = z,
                drift_intercept = drift_intercept,
                drift_slope = drift_slope,
                drift_angry = drift_angry,
                drift_neutral = drift_neutral
            ) 
        )
    end
    sort!(df, :subject_id)

    return df
end

function results_to_regressors(res::FacesResult, df; trial_duration=4, T_sample=0.4)
    subject_ID = only(res.subject_ID)
    run = only(res.run)
    session = only(res.session)

    df_fit = @subset(df, :subject_id .== subject_ID, :run .== run, :session .== session)
    df_agr = read_aggressiveness(df_fit; normalize=false)
    add_data!(df_fit, df_agr)

    df_regress = DataFrame(t = Float64[], evidence = Float64[])

    aggressiveness = df_fit.score
    RTs = get_response_times(df_fit)

    α, τ, z, drift_intercept, drift_slope = res.sol
    drifts = drift_intercept .+ drift_slope .* aggressiveness

    for (i, RT) in enumerate(RTs)
        t_regress = if RT > T_sample
            range(T_sample, RT, step = T_sample)
        else
            [RT]
        end

        for t in t_regress
            ev = drifts[i] * t
            t_sample = t + (i - 1) * trial_duration
            push!(
                df_regress, 
                (t = t_sample, evidence = ev)
            )
        end
    end

    CSV.write("faces_regress_sub-$(subject_ID)_ses-$(session)_run-$(run).csv", df_regress)
end
