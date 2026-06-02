function probability_choices(Delta_loglikelihoods::Float64, β, P_lapse)
    P_left = logistic(β * Delta_loglikelihoods)
    
    return [P_left, 1 - P_left] .* (1 - P_lapse) .+ P_lapse / 2
end

function probability_choices(loglikelihoods::AbstractMatrix, response_dots, β, P_lapse)
    z_left, z_right = sum(loglikelihoods[1:response_dots, :]; dims=1)
    Delta_loglikelihoods = z_left - z_right
    
    return probability_choices(Delta_loglikelihoods, β, P_lapse)
end

function negative_loglikelihood(β, P_lapse, dot_evidence::Vector{Matrix{Float64}}, choices::AbstractVector, response_dots::AbstractVector)
    N_trials = length(dot_evidence)

    loglikelihood = 0.0

    for t in Base.OneTo(N_trials)
        loglikelihoods = dot_evidence[t]
        RD = response_dots[t]
        
        P_choices = probability_choices(loglikelihoods, RD, β, P_lapse)
        choice_idx = choices[t] + 1
        P_choice = P_choices[choice_idx] 
       
        loglikelihood += log(P_choice)
    end

    return -loglikelihood
end

struct CMResult
    sol
    subject_id
    session
    run
end

function fit_CM(df)
    L = get_loglikelihood_dots(df)
    C = get_choices(df)
    RD = get_response_dots(df)

    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.WARN, linear_solver=LapackCPUSolver))
    @variable(model, 1e-4 <= β <= 100.0)
    @variable(model, 1e-4 <= P_lapse <= 1)
    @operator(model, neglhood, 2, (β, P_lapse) -> negative_loglikelihood(β, P_lapse, L, C, RD))
    @objective(model, Min, neglhood(β, P_lapse))

    optimize!(model)

    res = CMResult(model, unique(df.subject_id), unique(df.session), unique(df.run))

    return res
end

function CM_results_to_regressors(df_res, df_data; inter_dot_interval = 0.55)
    for r in eachrow(df_res)
        df_fit = @subset(df_data, :subject_id .== r.subject_id, :run .== r.run, :session .== r.session)
        
        df_regress = DataFrame(t = Float64[], P_chosen = Float64[], P_unchosen = Float64[], P_left = Float64[], P_right = Float64[])

        ST = parse.(Float64, df_fit.stim_presentation_time)
        RT = parse.(Float64, df_fit.response_time)

        RD = get_response_dots(df_fit)
        L = get_loglikelihood_dots(df_fit)
        C = get_choicesp1(df_fit)

        β = r.β
        P_lapse = r.P_lapse

        for t in eachindex(RT)
            for d in Base.OneTo(RD[t])
                t_dot = ST[t] + d*inter_dot_interval
                choice_cat = C[t]

                Ps = probability_choices(L[t], d, β, P_lapse)

                idx_unchosen = choice_cat == 2 ? 1 : 2
                push!(
                    df_regress, 
                    (t = t_dot, P_chosen = Ps[choice_cat], P_unchosen = Ps[idx_unchosen], P_left = Ps[1], P_right = Ps[2])
                )
            end
        end

        CSV.write("CM_regress_sub-$(r.subject_id)_ses-$(r.session)_run-$(r.run).csv", df_regress)
    end
end

function results_to_dataframe(results::Vector{<:CMResult})
    df = DataFrame(
        subject_id = Int64[],
        run = Int64[],
        session = String[],
        β = Float64[],
        P_lapse = Float64[]
    )

    for r in results
        push!(
            df,
            (
                subject_id = only(r.subject_id),
                run = only(r.run),
                session = only(r.session),
                β = value(variable_by_name(r.sol, "β")),
                P_lapse = value(variable_by_name(r.sol, "P_lapse"))
            ) 
        )
    end
    sort!(df, :subject_id)

    return df
end
