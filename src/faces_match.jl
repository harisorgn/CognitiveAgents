function negative_loglikelihood(α, τ, z, drift_intercept, drift_slope, data, aggressiveness)
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
    subject_id
    session
    run
end

function fit_faces(df; min_rt = 0.2, kwargs...)
    df_agr = read_aggressiveness(df; zero_center=true)
    add_data!(df, df_agr)

    filter!(:rt => rt -> !ismissing(rt) && rt >= min_rt, df)

    data = map(eachrow(df)) do r
        (choice = r.choice, rt = r.rt)
    end

    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.WARN, linear_solver=LapackCPUSolver))
    @variable(model, 0.5 <= α <= Inf)
    @variable(model, 1e-4 <= τ <= minimum(df.rt))
    @variable(model, 1e-4 <= z <= 1.0)
    @variable(model, -Inf <= drift_intercept <= Inf)
    @variable(model, -Inf <= drift_slope <= Inf)
    @operator(model, neglhood, 5, (α, τ, z, drift_intercept, drift_slope) -> negative_loglikelihood(α, τ, z, drift_intercept, drift_slope, data, df.score))
    @objective(model, Min, neglhood(α, τ, z, drift_intercept, drift_slope))

    optimize!(model)

    res = FacesResult(model, unique(df.subject_id), unique(df.session), unique(df.run))

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
        drift_neutral = Float64[],
        drift_ambiguous = Float64[]
    )

    for r in results
        α = value(variable_by_name(r.sol, "α"))
        τ = value(variable_by_name(r.sol, "τ")) 
        z = value(variable_by_name(r.sol, "z")) 
        drift_intercept = value(variable_by_name(r.sol, "drift_intercept")) 
        drift_slope = value(variable_by_name(r.sol, "drift_slope"))

        drift_angry = drift_intercept - 4.5 * drift_slope
        drift_neutral = drift_intercept + 4.5 * drift_slope
        drift_ambiguous = drift_intercept

        push!(
            df,
            (
                subject_id = only(r.subject_id),
                run = only(r.run),
                session = only(r.session),
                α = α,
                τ = τ,
                z = z,
                drift_intercept = drift_intercept,
                drift_slope = drift_slope,
                drift_angry = drift_angry,
                drift_neutral = drift_neutral,
                drift_ambiguous = drift_intercept
            ) 
        )
    end
    sort!(df, :subject_id)

    return df
end

function faces_results_to_regressors(df_res, df; trial_duration=4, T_sample=0.4)
    for r in eachrow(df_res)
        df_fit = @subset(df, :subject_id .== r.subject_id, :run .== r.run, :session .== r.session)
        df_agr = read_aggressiveness(df_fit; zero_center=true)
        add_data!(df_fit, df_agr)

        df_regress = DataFrame(t = Float64[], evidence = Float64[])

        aggressiveness = df_fit.score
        RTs = get_response_times(df_fit)

        α = r.α 
        τ = r.τ
        z = r.z 
        drift_intercept = r.drift_intercept 
        drift_slope = r.drift_slope
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

        CSV.write("faces_regress_sub-$(r.subject_id)_ses-$(r.session)_run-$(r.run).csv", df_regress)
    end
end
