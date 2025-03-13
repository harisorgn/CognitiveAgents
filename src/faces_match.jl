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


function results_to_dataframe(results)
    df = DataFrame(
        subject_ID = Int64[],
        run = Int64[],
        session = String[],
        α = Float64[],
        τ = Float64[],
        z = Float64[],
        drift_intercept = Float64[],
        drift_slope = Float64[],
        drift_angry = Float64[],
        drift_amb = Float64[],
        drift_neutral = Float64[]
    )

    for r in results
        run = if r.run isa Number
            r.run
        else
            if r.run .== "PRE"
                1
            elseif r.run .== "POST"
                2
            else
                parse(Int, r.run)
            end
        end

        N_samples = length(r.chain)

        drift_angry = r.chain[:drift_intercept].data .+ 1 .* r.chain[:drift_slope].data
        drift_neutral = r.chain[:drift_intercept].data .+ 10 .* r.chain[:drift_slope].data
        drift_amb = r.chain[:drift_intercept].data .+ 5 .* r.chain[:drift_slope].data # 5 is the aggressiveness value for the most ambiguous faces
        append!(
            df,
            (
                subject_ID = fill(r.subject_ID, N_samples),
                run = fill(run, N_samples),
                session = fill(r.session, N_samples),
                α = vec(r.chain[:α].data),
                τ = vec(r.chain[:τ].data),
                z = vec(r.chain[:z].data),
                drift_intercept = vec(r.chain[:drift_intercept].data),
                drift_slope = vec(r.chain[:drift_slope].data),
                drift_amb = vec(drift_amb),
                drift_angry = vec(drift_angry),
                drift_neutral = vec(drift_neutral)
            ) 
        )
    end
    sort!(df, :subject_ID)

    return df
end

function results_to_regressors(res::FacesResult, df)
    df_regress = DataFrame(t = Float64[], P_chosen = Float64[], P_unchosen = Float64[], P_left = Float64[], P_right = Float64[])

    aggressiveness = df.score

    RT = get_response_times(df)

    α, τ, z, drift_intercept, drift_slope = res.sol
    drifts = drift_intercept .+ drift_slope .* aggressiveness

    for t in eachindex(RT)
        push!(
            df_regress, 
            (t = t_dot, P_chosen = P[C[t]], P_unchosen = P[idx_unchosen], P_left = P[1], P_right = P[2])
        )
    end

    CSV.write("CM_regress_sub-$(res.subject_ID)_ses-$(res.session)_run-$(res.run).csv", df_regress)
end
