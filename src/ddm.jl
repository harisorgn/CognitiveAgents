@model function ddm(data, conditions, N_conditions; min_rt=0.2)

    α ~ LogNormal(1, 0.5)
    τ ~ Uniform(0.0, min_rt)
    z ~ Beta(1.5, 1.5)

    drift_intercept ~ Normal(0, 1)
    drift_slope ~ Normal(0, 1)

    drifts = drift_intercept .+ drift_slope .* conditions

    Turing.@addlogprob! sum(logpdf.(DDM.(drifts, α, τ, z), data))

end

function add_data!(df, df_aggressive)
    N_faces = nrow(df_aggressive)

    df.choice = choicesp1(df)
    df.rt = response_times(df)
    df.score = repeat(df_aggressive.score, Int(nrow(df) / N_faces))
end

struct FacesResult
    chain
    subject_ID
    session
    run
end

function fit_model(df, filename; min_rt = 0.2)
    
    add_data!(df, df_aggressive)

    filter!(:rt => rt -> !ismissing(rt) && rt >= min_rt, df)

    data = map(eachrow(df)) do r
        (choice = r.choice, rt = r.rt)
    end
    @assert all(map(d -> d.choice==1 || d.choice == 2, data))
    N_conditions = length(unique(df.score))

    md = ddm(data, df.score, N_conditions; min_rt)

    chain = sample(md, NUTS(), 2000)

    id = only(unique(df.subject_id))
    session = only(unique(df.session))
    run = only(unique(df.run))

    res = FacesResult(chain, id, session, run)

    serialize(string("./results/", filename, ".jls"), res)

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