function plot_subject_accuracy(df, N_trials_per_set=20, N_trials_average::Int=Int(round(N_trials_per_set/5)); name="", save_plot=false, title="")
    IDs = unique(df[!, :subject_id])    
    sets = unique(df[!, :set])

    colormap = ColorSchemes.seaborn_bright.colors

    N_points_per_set = N_trials_per_set / N_trials_average
    xlabel_ticks = (N_points_per_set/2):N_points_per_set:(N_points_per_set * length(sets))
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        ylabel = "Accuracy",
        xticks = (xlabel_ticks , ["Set 1", "Set 2", "Set 3", "Set 4"])
    )
    hidexdecorations!(ax, ticklabels = false)

    for (i,ID) in enumerate(IDs)
        xs = collect(1:N_points_per_set)
        for (j,s) in enumerate(sets)
            choices = parse.(Bool, df[(df.subject_id .== ID) .& (df.set .== s), :correct])
            acc = mean.(partition(choices, N_trials_average))
        
            scatter!(ax, xs, acc, color=(colormap[mod(i, length(colormap))+1], 0.3))
            lines!(ax, xs, acc, color=(colormap[mod(i, length(colormap))+1], 0.3))
            
            xs .+= N_points_per_set
        end
    end

    vlines!(ax, collect(N_points_per_set:N_points_per_set:(N_points_per_set * length(sets))), linestyle = :dash, linewidth = 2, color=:gray)
    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)

    if save_plot
        save(string("subject_acc_", name, ".png"), f, pt_per_unit=1)
    end

    f
end

function group_accuracy_per_set!(ax, df, N_trials_per_set)
    IDs = unique(df[!, :subject_id])    
    sets = unique(df[!, :set])

    RESP = []
    for (i,s) in enumerate(sets)
        corrects = []
        for ID in IDs
            c = df[(df.subject_id .== ID) .& (df.set .== s), :correct]
            if length(c) == N_trials_per_set
                push!(corrects, parse.(Bool, c))
            end
        end
        if !isempty(corrects)
            push!(RESP, reduce(hcat, corrects))
        end
    end

    A = [vec(mean(R; dims=2)) for R in RESP]

    xs = collect(1:N_trials_per_set)

    for acc in A
        scatter!(ax, xs, acc, color=:black)
        lines!(ax, xs, acc, color=(:black, 1))
        vlines!(ax, [last(xs)], linestyle = :dash, linewidth = 2, color=:gray)
        xs .+= N_trials_per_set
    end
end

function plot_group_accuracy_per_set(df, N_trials_per_set; name="", save_plot=false, title="", N_training=0)
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Trial",
        ylabel = "Accuracy"
    )

    group_accuracy_per_set!(ax, df, N_trials_per_set)

    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)

    if save_plot
        save(string("group_acc_per_set_", name, ".png"), f, pt_per_unit=1)
    end

    f
end

function group_accuracy!(ax, df, N_trials_per_set; color=:black, kwargs...)
    IDs = unique(df[!, :subject_id])    
    sets = unique(df[!, :set])
    N_subjects = length(IDs)

    RESP = []
    for (i, s) in enumerate(sets)
        corrects = Matrix{Union{Missing, Bool}}(undef, N_subjects, N_trials_per_set)
        for (j, ID) in enumerate(IDs)
            c = df[(df.subject_id .== ID) .& (df.set .== s), :correct]
            corrects[j, 1:length(c)] = parse.(Bool, c)
        end
        push!(RESP, corrects)
    end
    
    M = mapreduce(hcat, RESP) do R
        map(eachcol(R)) do R_trial
            mean(skipmissing(R_trial))
        end
    end
    xs = collect(1:N_trials_per_set)

    for (i, acc) in enumerate(eachcol(M))
        #scatter!(ax, xs, acc, color=(colormap[mod(i, length(colormap))+1], 0.3))
        #lines!(ax, xs, acc, color=(colormap[mod(i, length(colormap))+1], 0.3))
    end

    μ_group = vec(mean(M; dims=2))
    σ_group = vec(std(M; dims=2)) ./ sqrt(N_subjects)

    scatter!(ax, xs, μ_group; color=(color, 1))
    lines!(ax, xs, μ_group, color=(color, 1); kwargs...)
    band!(ax, xs, μ_group .- σ_group, μ_group .+ σ_group, color=(color, 0.3))

end

function plot_group_accuracy(df, N_trials_per_set=20; name="", label="", save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Trial",
        ylabel = "Accuracy",
        yticks = 0.0:0.2:1.0
    )

    group_accuracy!(ax, df, N_trials_per_set; color=colormap[1], label)

    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)
    ylims!(ax, 0, 1)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save(string("group_acc_", name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_group_accuracy(df, N_trials_per_set, session; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Trial",
        ylabel = "Accuracy",
        yticks = 0.0:0.2:1.0
    )

    group_accuracy!(
        ax, 
        df[(df.session.==session) .& (df.run.==1),:], 
        N_trials_per_set; 
        color=colormap[1],
        label = "Control"
    )

    group_accuracy!(
        ax, 
        df[(df.session.==session) .& (df.run.==2),:], 
        N_trials_per_set; 
        color=colormap[2],
        label = session.=="glc" ? "Glucose" : "BHB"
    )

    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)
    ylims!(ax, 0, 1)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save(string("task1_new_acc_$(session)", ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_RT!(ax, df; color=:black)
    RT = get_response_times(df)

    hist!(ax, RT; bins=50, color = (color, 0.3))
    vlines!(ax, median(RT); linestyle = :dash, linewidth = 2, color)
end

function plot_rt(df; name="", save_plot=false, title="", N_training=0)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Response time [sec]",
        xticks = 0.0:0.5:10
    )

    plot_RT!(ax, df; color=colormap[1])

    if save_plot
        save(string("RT_", name, ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_model_params(res, labels; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors

    N_subjects, N_conditions = size(res)

    f = Figure(;size = (1024, 768))
    ax = [
        Axis(
            f[1, 1], 
            title = "β : inverse temperature",
            xticks = ([1,2], labels)
        ),
        Axis(
            f[1, 2], 
            title = "σ : inference noise",
            xticks = ([1,2], labels)
        )
    ]        

    ylims!.(ax, 0, 4)

    for i in Base.OneTo(N_subjects)
        β = map(x -> x.β, res[i, :])
        σ = map(x -> x.σ_inf, res[i, :])

        scatter!(ax[1], 1:N_conditions, β; color=(colormap[mod(i, length(colormap))+1], 0.8))
        lines!(ax[1], 1:N_conditions, β; color=(colormap[mod(i, length(colormap))+1], 0.8))

        scatter!(ax[2], 1:N_conditions, σ; color=(colormap[mod(i, length(colormap))+1], 0.8))
        lines!(ax[2], 1:N_conditions, σ; color=(colormap[mod(i, length(colormap))+1], 0.8))
    end        

    if save_plot
        save(string(join(labels, "_VS_"), ".png"), f, pt_per_unit=1)
    end

    f
end


function plot_prob_choice(L, β, σ; save_plot=false)
    N_dots, N_categories = size(L)

    P = zeros(N_dots, N_categories)

    for dot in Base.OneTo(N_dots)
        evidence = sum(L[1:dot, :]; dims=1)

        samples = rand(MvNormal(vec(evidence), dot*σ), 10_000)

        Ps = softmax(β*samples)
        P[dot, :] = mean(Ps; dims=2)
    end

    f = Figure(;size = (1024, 768), fontsize=26)
    ax = Axis(
        f[1, 1], 
        xlabel = "Dot",
        xticks = 1:N_dots,
        ylabel = "P(choice | dots)"
    )

    stairs!(ax, 1:N_dots, P[:,1]; step=:post, label="Left")
    stairs!(ax, 1:N_dots, P[:,2]; step=:post, label="Right")

    f[1, 2] = Legend(f, ax, "Choice", framevisible = false)

    if save_plot
        save("example_prob_choice.png", f, pt_per_unit=1)
    end

    f
end

function plot_cumulative_RT!(ax, df, xs; color=:black, kwargs...)
    RT = df.response_time
    filter!(r -> r != "None", RT)
    RT = parse.(Float64, RT)
    
    f = ecdf(RT)
    
    lines!(ax, xs, f.(xs); color = color, kwargs...)
    vlines!(ax, median(RT); linestyle = :dash, linewidth = 2, color)
end

function figure_cumulative_RT(df, session, xs; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Response time [sec]",
        xticks = 0.0:0.5:10,
        ylabel = "Cumulative Probability"
    )

    plot_cumulative_RT!(
        ax, 
        df[(df.session.==session) .& (df.run.==1),:], 
        xs; 
        color=colormap[1],
        label = "Control"
    )

    plot_cumulative_RT!(
        ax, 
        df[(df.session.==session) .& (df.run.==2),:], 
        xs; 
        color=colormap[2],
        label = session.=="glc" ? "Glucose" : "BHB"
    )

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save(string("task2_cRT_$(session)", ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_RT_faces!(ax, df; color=:black, kwargs...)
    RT = get_response_times(df)
    aggressiveness = df.score
    scores = sort(unique(aggressiveness))

    μ_RT_score = map(scores) do s
        idx = aggressiveness .== s
        mean(skipmissing(RT[idx]))
    end

    sem_RT_score = map(scores) do s
        idx = aggressiveness .== s
        std(skipmissing(RT[idx])) / sqrt(count(.!ismissing.(RT[idx])))
    end

    lines!(ax, scores, μ_RT_score; color, kwargs...)
    scatter!(ax, scores, μ_RT_score; color)
    errorbars!(ax, scores, μ_RT_score, sem_RT_score; color)
end

function plot_RT_faces!(ax, res::FacesResult, scores; color=:black, kwargs...)
    α, τ, z, drift_intercept, drift_slope = res.sol
    N_samples = 10_000

    μ_RT_score = map(scores) do s
        drift = drift_intercept + (drift_slope * s)
        model = DDM(drift, α, z, τ)
        RT = rand(model, N_samples).rt
        mean(RT)
    end

    sem_RT_score = map(scores) do s
        drift = drift_intercept + (drift_slope * s)
        model = DDM(drift, α, z, τ)
        RT = rand(model, N_samples).rt
        std(RT) / sqrt(N_samples)
    end

    lines!(ax, scores, μ_RT_score; color, kwargs...)
    scatter!(ax, scores, μ_RT_score; color)
    errorbars!(ax, scores, μ_RT_score, sem_RT_score; color)
end

function figure_faces_RT(df::DataFrame, session; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Aggressiveness",
        xticks = 0:10,
        ylabel = "Response Time [sec]"
    )

    plot_RT_faces!(
        ax, 
        df[(df.session.==session) .& (df.run.==1),:]; 
        color=colormap[1],
        label = "Control"
    )

    plot_RT_faces!(
        ax, 
        df[(df.session.==session) .& (df.run.==2),:]; 
        color=colormap[2],
        label = session.=="glc" ? "Glucose" : "BHB"
    )
 
    image!(ax, (0.5,1.5), (0.7,0.85), rotr90(load("./figures/angry.png")))
 
    image!(ax, (4.5,5.5), (0.7,0.85), rotr90(load("./figures/ambiguous.png")))

    image!(ax, (9.5,10.5), (0.7,0.85), rotr90(load("./figures/neutral.png")))

    #ylims!.(ax, 0.6, 1.6)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save("faces_RT_ses_$(session).png", f, pt_per_unit=1)
    end

    f
end

function figure_faces_RT(df::DataFrame, res::FacesResult; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Aggressiveness",
        xticks = (-4.5:4.5, ["$(i)" for i in Base.OneTo(10)]),
        ylabel = "Response Time [sec]"
    )

    plot_RT_faces!(ax, df; color=colormap[1], label = "Data")

    scores = sort(unique(df.score))
    plot_RT_faces!(ax, res, scores; color=colormap[2], label = "Model")
 
    #image!(ax, (0.5,1.5), (0.7,0.85), rotr90(load("./figures/angry.png")))

    #image!(ax, (4.5,5.5), (0.7,0.85), rotr90(load("./figures/ambiguous.png")))

    #image!(ax, (9.5,10.5), (0.7,0.85), rotr90(load("./figures/neutral.png")))

    #ylims!.(ax, 0.6, 1.6)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save("faces_RT_model.png", f, pt_per_unit=1)
    end

    f
end

function plot_faces_psychophysics!(ax, df::DataFrame; color=:black, kwargs...)
    choices = get_choicesp1(df)
    aggressiveness = df.score
    scores = sort(unique(aggressiveness))

    μ_acc_score = map(scores) do s
        idx = aggressiveness .== s
        N_trials_score = sum(idx)
        sum(choices[idx] .== 2) / N_trials_score
    end

    sem_acc_score = map(scores) do s
        idx = aggressiveness .== s
        N_trials_score = sum(idx)
        std(choices[idx] .== 2) / sqrt(N_trials_score)
    end

    lines!(ax, scores, μ_acc_score; color, kwargs...)
    scatter!(ax, scores, μ_acc_score; color)
    errorbars!(ax, scores, μ_acc_score, sem_acc_score; color)
end

function plot_faces_psychophysics!(ax, res::FacesResult, scores; color=:black, kwargs...)
    α, τ, z, drift_intercept, drift_slope = res.sol
    N_samples = 10_000

    μ_acc_score = map(scores) do s
        drift = drift_intercept + (drift_slope * s)
        model = DDM(drift, α, z, τ)
        choices = rand(model, N_samples).choice
        sum(choices .== 2) / N_samples
    end

    sem_acc_score = map(scores) do s
        drift = drift_intercept + (drift_slope * s)
        model = DDM(drift, α, z, τ)
        choices = rand(model, N_samples).choice
        std(choices .== 2) / sqrt(N_samples)
    end

    lines!(ax, scores, μ_acc_score; color, kwargs...)
    scatter!(ax, scores, μ_acc_score; color)
    errorbars!(ax, scores, μ_acc_score, sem_acc_score; color)
end

function figure_faces_psychophysics(df::DataFrame, session; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    df_agr = read_aggressiveness(df; normalize=true)
    add_data!(df, df_agr)

    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Aggressiveness",
        xticks = (-4.5:4.5, ["$(i)" for i in Base.OneTo(10)]),
        yticks = 0:0.2:1,
        ylabel = "Probability of friend response"
    )

    plot_acc_aggressiveness!(
        ax, 
        df[(df.session.==session) .& (df.run.==1),:], 
        aggressiveness; 
        color=colormap[1],
        label = "Control"
    )

    plot_acc_aggressiveness!(
        ax, 
        df[(df.session.==session) .& (df.run.==2),:], 
        aggressiveness; 
        color=colormap[2],
        label = session.=="glc" ? "Glucose" : "BHB"
    )
 
    image!(ax, (0.5,1.5), (-0.25,-0.05), rotr90(load("./figures/angry.png")))
 
    image!(ax, (4.5,5.5), (-0.25,-0.05), rotr90(load("./figures/ambiguous.png")))

    image!(ax, (9.5,10.5), (-0.25,-0.05), rotr90(load("./figures/neutral.png")))

    ylims!.(ax, -0.3, 1.0)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save(string("faces_psychophysics_ses_$(session)", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_faces_psychophysics(df::DataFrame, res::FacesResult; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    df_agr = read_aggressiveness(df)
    add_data!(df, df_agr)

    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = "",
        xlabel = "Aggressiveness",
        xticks = (-4.5:4.5, ["$(i)" for i in Base.OneTo(10)]),
        yticks = 0:0.2:1,
        ylabel = "Probability of friend response"
    )

    plot_faces_psychophysics!(ax, df; color=colormap[1], label = "Data")

    scores = sort(unique(df.score))
    plot_faces_psychophysics!(ax, res, scores; color=colormap[2], label = "Model")
    #image!(ax, (0.5,1.5), (-0.25,-0.05), rotr90(load("./figures/angry.png")))
 
    #image!(ax, (4.5,5.5), (-0.25,-0.05), rotr90(load("./figures/ambiguous.png")))

    #image!(ax, (9.5,10.5), (-0.25,-0.05), rotr90(load("./figures/neutral.png")))

    ylims!.(ax, -0.3, 1.0)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_plot
        save("faces_psychophysics_model.png", f, pt_per_unit=1)
    end

    f
end

function plot_param_diff!(ax, df, param, sessions; colormap = ColorSchemes.seaborn_bright.colors)
    IDs = unique(df.subject_ID)
    runs = unique(df.run)

    @assert length(runs) == 2 

    for (i, ID) in enumerate(IDs)
        Δp = map(enumerate(sessions)) do (j,s)
            only(df[(df.subject_ID .== ID) .& (df.session .== s) .& (df.run .== 2), param]) - only(df[(df.subject_ID .== ID) .& (df.session .== s) .& (df.run .== 1), param])
        end
        scatter!(ax, 1:length(sessions), Δp; color = color=(colormap[mod(i, length(colormap))+1]))
        lines!(ax, 1:length(sessions), Δp; color = color=(colormap[mod(i, length(colormap))+1]))
    end
end

function figure_CL_model(df; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    sessions = unique(df.session)

    labels = map(sessions) do s
        if s == "glc"
            "Glucose"
        elseif s == "bhb"
            "BHB"
        else
            s
        end
    end

    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Feedback sensitivity",
                xticklabelsize = 20,
                yticklabelsize = 20
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Prototype learning rate",
                xticklabelsize = 20,
                yticklabelsize = 20
            ),
            Axis(
                f[1, 3], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Propensity to change rules",
                xticklabelsize = 20,
                yticklabelsize = 20
            )
    ]

    plot_param_diff!(ax[1], df, :η, sessions; colormap)
    plot_param_diff!(ax[2], df, :ηₓ, sessions; colormap)
    plot_param_diff!(ax[3], df, :α, sessions; colormap)

    colgap!(f.layout, 40)
    xlims!.(ax, 0.8, 2.2)

    supertitle = f[0, :] = Label(f, "Model parameter changes compared to baseline",
        fontsize = 30, color = (:black, 0.6))

    if save_plot
        save(string("task1_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_CM_model(df; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    sessions = unique(df.session)

    labels = map(sessions) do s
        if s == "glc"
            "Glucose"
        elseif s == "bhb"
            "BHB"
        else
            s
        end
    end

    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Inverse temperature",
                xticklabelsize = 20,
                yticklabelsize = 20
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Evidence accumulation noise",
                xticklabelsize = 20,
                yticklabelsize = 20
            )
    ]

    plot_param_diff!(ax[1], df, :β, sessions; colormap)
    plot_param_diff!(ax[2], df, :σ_inf, sessions; colormap)
    
    colgap!(f.layout, 40)
    xlims!.(ax, 0.8, 2.2)
    ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Model parameter changes compared to baseline",
        fontsize = 30, color = (:black, 0.6))

    if save_plot
        save(string("task2_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_posterior_param_diff!(ax, df, param, sessions; colormap = ColorSchemes.seaborn_bright.colors)
    IDs = unique(df.subject_ID)
    runs = unique(df.run)

    @assert length(runs) == 2 

    for (i, ID) in enumerate(IDs)
        Δp = map(enumerate(sessions)) do (j,s)
            @show ID
            @show s
            @show param
            df[(df.subject_ID .== ID) .& (df.session .== s) .& (df.run .== 2), param] .- df[(df.subject_ID .== ID) .& (df.session .== s) .& (df.run .== 1), param]
        end
        offset = rand(Normal(0, 0.1))

        μ = mean.(Δp)
        σ = std.(Δp)
        xs = (1:length(sessions)) .+ offset
        scatter!(ax, xs, μ;  color = (colormap[mod(i, length(colormap))+1]))
        errorbars!(ax, xs, μ, σ; color = (colormap[mod(i, length(colormap))+1]))
    end
end

function figure_faces_model(df; save_plot=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    sessions = unique(df.session)

    labels = map(sessions) do s
        if s == "glc"
            "Glucose"
        elseif s == "bhb"
            "BHB"
        else
            s
        end
    end

    fontsize_label = 18

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Starting point",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Bound seperation",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            ),
            Axis(
                f[1, 3], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Non-decision time",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            ),
            Axis(
                f[2, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Drift [Ambiguous]",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            ),
            Axis(
                f[2, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Drift [Angry]",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            ),
            Axis(
                f[2, 3], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Drift [Neutral]",
                xticklabelsize = fontsize_label,
                yticklabelsize = fontsize_label
            )
            
    ]

    plot_posterior_param_diff!(ax[1], df, :z, sessions; colormap)
    plot_posterior_param_diff!(ax[2], df, :α, sessions; colormap)
    plot_posterior_param_diff!(ax[3], df, :τ, sessions; colormap)
    plot_posterior_param_diff!(ax[4], df, :drift_amb, sessions; colormap)
    plot_posterior_param_diff!(ax[5], df, :drift_angry, sessions; colormap)
    plot_posterior_param_diff!(ax[6], df, :drift_neutral, sessions; colormap)

    colgap!(f.layout, 40) 
    xlims!.(ax, 0.7, 2.3)
    #ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Model parameter changes compared to baseline",
        fontsize = 26, color = (:black, 0.6))

    if save_plot
        save(string("task3_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_CM_psychophysics(df, β, P_lapse)
    L = get_loglikelihood_dots(df)
    C = get_choices(df)
    RD = get_response_dots(df)

    z_left = map(enumerate(L)) do (t, loglikelihoods)
        sum(loglikelihoods[1:RD[t], 1])
    end

    z_right = map(enumerate(L)) do (t, loglikelihoods)
        sum(loglikelihoods[1:RD[t], 2])
    end

    Delta_llhood = z_right .- z_left

    edges = range(minimum(Delta_llhood), maximum(Delta_llhood); length=11)
    
    P_right_choices = map(eachindex(edges[1:end-1])) do i
        idx = findall(Delta_llhood) do Dl
            (Dl >= edges[i]) && (Dl < edges[i+1])
        end

        sum(C[idx]) / length(C[idx])
    end

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = Axis(f[1,1])

    scatter!(ax, edges[1:end-1], P_right_choices; color=:blue)
    
    r = range(minimum(Delta_llhood), maximum(Delta_llhood); length=1000)
    
    P_right_model = last.(probability_choices.(-r, β, P_lapse))

    lines!(ax, r, P_right_model; color=:orange)

    f
end
