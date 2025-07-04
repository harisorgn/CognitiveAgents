function plot_subject_accuracy!(ax::Axis, df::DataFrame, N_trials_per_set, N_trials_average; label="", kwargs...)
    sets = unique(df[!, :set])
    N_points = N_trials_per_set / N_trials_average

    xs = collect(1:N_points)
    for s in sets
        corrects = get_corrects(df[df.set .== s, :])
        acc = mean.(partition(corrects, N_trials_average))
    
        scatter!(ax, xs, acc; kwargs... )
        lines!(ax, xs, acc; label, kwargs...)
        
        xs .+= N_points
    end
end

function plot_subject_accuracy!(ax::Axis, gdf::GroupedDataFrame, N_trials_per_set, N_trials_average; label="", kwargs...)
    sets = unique(combine(gdf, :set).set)
    N_points = N_trials_per_set / N_trials_average

    df = combine(gdf, :correct => mean => :accuracy, :set => only ∘ unique => :set)

    xs = collect(1:N_points)
    for s in sets
        acc = mean.(partition(df[df.set .== s, :accuracy], N_trials_average))
    
        scatter!(ax, xs, acc; kwargs... )
        lines!(ax, xs, acc; label, kwargs...)
        
        xs .+= N_points
    end
end

function figure_subject_accuracy(df; N_trials_per_set=20, N_trials_average::Int=Int(round(N_trials_per_set/5)), name="", save=false, title="")
    colormap = ColorSchemes.seaborn_bright.colors

    sets = unique(df[!, :set])

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

    gdf = groupby(df, :subject_id)
    for (i,df_subj) in enumerate(gdf)
        color = (colormap[mod(i, length(colormap))+1], 0.3)
        plot_subject_accuracy!(ax, df_subj, N_trials_per_set, N_trials_average; color)
    end

    vlines!(ax, collect(N_points_per_set:N_points_per_set:(N_points_per_set * length(sets))), linestyle = :dash, linewidth = 2, color=:gray)
    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)

    if save
        save(string("CL_subject_acc_", name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_subject_accuracy(df, res::CLResult; N_trials_per_set=20, N_trials_average::Int=Int(round(N_trials_per_set/5)), name="", save_fig=false, title="")
    colormap = ColorSchemes.seaborn_bright.colors

    sets = unique(df[!, :set])

    N_points_per_set = N_trials_per_set / N_trials_average
    xlabel_ticks = (N_points_per_set/2):N_points_per_set:(N_points_per_set * length(sets))
    
    f = Figure(;size = (1024, 768), fontsize=26)
    ax = Axis(
        f[1, 1], 
        title = title,
        ylabel = "Accuracy",
        xticks = (xlabel_ticks , ["Set 1", "Set 2", "Set 3", "Set 4"])
    )
    hidexdecorations!(ax, ticklabels = false)

    plot_subject_accuracy!(ax, df, N_trials_per_set, N_trials_average; color = colormap[1], label = "Data")

    df_sim = run_CL_task(df, res; N_runs = 100)
    gdf_sim = groupby(df_sim, :trial_index)
    plot_subject_accuracy!(ax, gdf_sim, N_trials_per_set, N_trials_average; color = colormap[2], label = "Model")

    vlines!(ax, collect(N_points_per_set:N_points_per_set:(N_points_per_set * length(sets))), linestyle = :dash, linewidth = 2, color=:gray)
    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)

    f[1, 2] = Legend(f, ax, framevisible = false, unique = true)

    if save_fig
        save(string("CL_subject_acc.png"), f, pt_per_unit=1)
    end

    f
end

function plot_group_accuracy!(ax::Axis, gdf::GroupedDataFrame, N_trials_per_set, N_subjects; label="", colormap=ColorSchemes.seaborn_bright.colors)
    sets = unique(combine(gdf, :set).set)
    N_points = N_trials_per_set
    
    df = combine(
        gdf, 
        :correct => mean => :accuracy, 
        :correct => (c -> std(c) ./ N_subjects) => :sem_accuracy,
        :set => only ∘ unique => :set
    )
    
    xs = collect(1:N_points)
    for s in sets
        acc = df[df.set .== s, :accuracy]
        sem_acc = df[df.set .== s, :sem_accuracy]

        scatter!(ax, xs, acc; color = colormap[1])
        lines!(ax, xs, acc; label, color = colormap[1])
        band!(ax, xs, acc .- sem_acc, acc .+ sem_acc; color = (colormap[2], 0.3))

        xs .+= N_points
    end
end

function figure_group_accuracy(df ; N_trials_per_set=20, name="CL_group_acc", save_fig=false, title="")
    colormap = ColorSchemes.seaborn_bright.colors

    sets = unique(df[!, :set])

    N_points_per_set = N_trials_per_set
    xlabel_ticks = (N_points_per_set/2):N_points_per_set:(N_points_per_set * length(sets))
    
    f = Figure(;size = (1024, 768), fontsize=26)
    ax = Axis(
        f[1, 1], 
        title = title,
        ylabel = "Accuracy",
        xticks = (xlabel_ticks , ["Set $(s)" for s in sets])
    )
    hidexdecorations!(ax, ticklabels = false)

    N_subjects = length(unique(df[!, :subject_id]))  
    gdf = groupby(df, :trial_index)
    plot_group_accuracy!(ax, gdf, N_trials_per_set, N_subjects; colormap)

    vlines!(ax, collect(N_points_per_set:N_points_per_set:(N_points_per_set * length(sets))), linestyle = :dash, linewidth = 2, color=:gray)
    hlines!(ax, [0.5], linestyle = :dash, linewidth = 2, color=:grey)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_cumulative_RT!(ax::Axis, gdf::GroupedDataFrame, xs; colormap=ColorSchemes.seaborn_bright.colors)
    for (i, df_subj) in enumerate(gdf) 
        RT = df_subj.response_time
        filter!(r -> r != "None", RT)
        RT = parse.(Float64, RT)
        
        f = ecdf(RT)
        
        lines!(ax, xs, f.(xs); label, color = colormap[mod(i, length(colormap))+1])
    end
end

function figure_cumulative_RT(df, xlims=(0,10); save_fig=false, name="", title="")
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Response time [sec]",
        xticks = first(xlims):last(xlims),
        ylabel = "Cumulative Probability"
    )

    gdf = groupby(df, :subject_id)
    xs = first(xlims):0.001:last(xlims)
    plot_cumulative_RT!(ax, gdf, xs; colormap)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_RT!(ax::Axis, gdf::GroupedDataFrame, edges; color=:black, bin_size=0.5)
    
    COUNTS = mapreduce(hcat, enumerate(gdf)) do (i, df)
        RT = get_response_times(df)
        filter!(r -> !ismissing(r), RT)

        h = fit(Histogram, RT, edges)
        h_n = normalize(h; mode=:pdf)
        h_n.weights
    end
    
    N_subjects = length(gdf)
    μ_counts = vec(mean(COUNTS; dims=2))
    sem_counts = vec(std(COUNTS; dims=2) ./ sqrt(N_subjects))

    xs = zeros(length(edges) - 1)
    step = 0
    for i in eachindex(edges)[2:end]
        xs[i-1] = step + (edges[i] - edges[i-1]) / 2
        step += bin_size
    end
    
    barplot!(ax, xs, μ_counts; gap=0, strokecolor=color, strokewidth=1, color=(color, 0.5))
    errorbars!(ax, xs, μ_counts, sem_counts; color, whiskerwidth = 12)
end

function figure_RT(df; bin_size=0.5, save_fig=false, name="", title="")
    colormap = ColorSchemes.seaborn_bright.colors
    
    edges = 0:bin_size:6

    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Response time [sec]",
        ylabel = "Probability density function",
        xticks = edges
    )
    
    gdf = groupby(df, :subject_id)
    plot_RT!(ax, gdf, edges; color=colormap[1], bin_size)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_RT_faces!(ax::Axis, df::Union{DataFrame, SubDataFrame}; color)
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

    barplot!(ax, scores, μ_RT_score; color=(color, 0.5))
    errorbars!(ax, scores, μ_RT_score, sem_RT_score; color, whiskerwidth = 12)
end

function plot_RT_faces!(ax::Axis, gdf::GroupedDataFrame; colormap=ColorSchemes.seaborn_bright.colors)
    for (i, df) in enumerate(gdf)
        plot_RT_faces!(ax, df; color = colormap[mod(i, length(colormap))+1])
    end
end

function plot_RT_faces!(ax, res::FacesResult, scores; colormap=ColorSchemes.seaborn_bright.colors)
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

    barplot!(ax, scores, μ_RT_score; color=(color, 0.5))
    errorbars!(ax, scores, μ_RT_score, sem_RT_score; color, whiskerwidth = 12)
end

function figure_RT_faces(df::DataFrame ; save_fig=false, name="", title="")
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Aggressiveness",
        xticks = 0:10,
        ylabel = "Response Time [sec]"
    )

    df_agr = read_aggressiveness(df; normalize=false)
    add_data!(df, df_agr)
    
    plot_RT_faces!(ax, df; color=colormap[1])
 
    image!(ax, (0.5,1.5), (-0.3, -0.05), rotr90(load("./stimuli/face_angry.png")))
 
    image!(ax, (4.5,5.5), (-0.3, -0.05), rotr90(load("./stimuli/face_ambiguous.png")))

    image!(ax, (9.5,10.5), (-0.3, -0.05), rotr90(load("./stimuli/face_neutral.png")))

    ylims!.(ax, -0.3, 1.6)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_RT_faces(df::DataFrame, res::FacesResult; save_fig=false)
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
 
    image!(ax, (0.5,1.5), (0.7,0.85), rotr90(load("./stimuli/face_angry.png")))
 
    image!(ax, (4.5,5.5), (0.7,0.85), rotr90(load("./stimuli/face_ambiguous.png")))

    image!(ax, (9.5,10.5), (0.7,0.85), rotr90(load("./stimuli/face_neutral.png")))

    #ylims!.(ax, 0.6, 1.6)

    f[1, 2] = Legend(f, ax, framevisible = false)

    if save_fig
        save("faces_RT_model.png", f, pt_per_unit=1)
    end

    f
end

function plot_psychophysics_faces!(ax, df::DataFrame; color=:black)
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

    lines!(ax, scores, μ_acc_score; color)
    scatter!(ax, scores, μ_acc_score; color)
    errorbars!(ax, scores, μ_acc_score, sem_acc_score; color)
end

function plot_psychophysics_faces!(ax, res::FacesResult, scores; color=:black)
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

    lines!(ax, scores, μ_acc_score; color)
    scatter!(ax, scores, μ_acc_score; color)
    errorbars!(ax, scores, μ_acc_score, sem_acc_score; color)
end

function figure_psychophysics_faces(df::DataFrame; save_fig=false, name="", title="")
    colormap = ColorSchemes.seaborn_bright.colors

    f = Figure(;size = (1024, 768), fontsize=30)
    ax = Axis(
        f[1, 1], 
        title = title,
        xlabel = "Aggressiveness",
        xticks = Base.OneTo(10),
        yticks = 0:0.2:1,
        ylabel = "Probability of friend response"
    )

    df_agr = read_aggressiveness(df; normalize=false)
    add_data!(df, df_agr)

    plot_psychophysics_faces!(ax, df; color=colormap[1])
 
    image!(ax, (0.5,1.5), (-0.25,-0.05), rotr90(load("./stimuli/face_angry.png")))
 
    image!(ax, (4.5,5.5), (-0.25,-0.05), rotr90(load("./stimuli/face_ambiguous.png")))

    image!(ax, (9.5,10.5), (-0.25,-0.05), rotr90(load("./stimuli/face_neutral.png")))

    ylims!.(ax, -0.3, 1.0)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_psychophysics_faces(df::DataFrame, res::FacesResult; save_fig=false)
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

    if save_fig
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

function figure_CL_model(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors

    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Feedback sensitivity",
                xticklabelsize = 26,
                yticklabelsize = 26
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Exploration/Exploitation",
                xticklabelsize = 26,
                yticklabelsize = 26
            )
    ]

    df_control = @subset(df, :subject_id .<= 99, :run .== 1)
    df_bipolar = @subset(df, :subject_id .> 99, :run .== 1)

    scatter!(ax[1], fill(1, nrow(df_control)), df_control.η; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar)), df_bipolar.η; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control)), df_control.β; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar)), df_bipolar.β; color=colormap[2])

     xlims!.(ax, 0.8, 2.2)

    supertitle = f[0, :] = Label(f, "Dot Category Learn model parameters",
        fontsize = 30, color = (:black, 0.6))

    if save
        save(string("task1_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_CL_model_param_diff(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors

    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Feedback sensitivity",
                xticklabelsize = 26,
                yticklabelsize = 26
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Exploration/Exploitation",
                xticklabelsize = 26,
                yticklabelsize = 26
            )
    ]

    df_control_pre = @subset(df, :subject_id .<= 99, :run .== 1, :session .== "bhb")
    df_control_post = @subset(df, :subject_id .<= 99, :run .== 2, :session .== "bhb")
    df_bipolar_pre = @subset(df, :subject_id .> 99, :run .== 1, :session .== "bhb")
    df_bipolar_post = @subset(df, :subject_id .> 99, :run .== 2, :session .== "bhb")

    scatter!(ax[1], fill(1, nrow(df_control_pre)), df_control_post.η .- df_control_pre.η; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.η .- df_bipolar_pre.η; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control_pre)), df_control_post.β .- df_control_pre.β; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.β .- df_bipolar_pre.β; color=colormap[2])

    xlims!.(ax, 0.8, 2.2)

    supertitle = f[0, :] = Label(f, "Dot Category Learn model parameter differences",
        fontsize = 30, color = (:black, 0.6))

    if save
        save(string("task1_model_param_diff", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_CM_model(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Evidence accumulation noise",
                xticklabelsize = 26,
                yticklabelsize = 26
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Exploration/Exploitation",
                xticklabelsize = 26,
                yticklabelsize = 26
            )
    ]

    df_control = @subset(df, :subject_id .<= 99, :run .== 1)
    df_bipolar = @subset(df, :subject_id .> 99, :run .== 1)

    scatter!(ax[1], fill(1, nrow(df_control)), df_control.σ_inf; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar)), df_bipolar.σ_inf; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control)), df_control.β; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar)), df_bipolar.β; color=colormap[2])

    colgap!(f.layout, 40)
    xlims!.(ax, 0.8, 2.2)
    ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Dot Category Match model parameters",
        fontsize = 30, color = (:black, 0.6))

    if save
        save(string("task1_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_CM_model_param_diff(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors
    
    f = Figure(;size = (1280, 720), fontsize=30)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Evidence accumulation noise",
                xticklabelsize = 26,
                yticklabelsize = 26
            ),
            Axis(
                f[1, 2], 
                title = "",
                xlabel = "",
                xticks = ([1,2], ["Control", "Bipolar"]),
                ylabel = "Exploration/Exploitation",
                xticklabelsize = 26,
                yticklabelsize = 26
            )
    ]

    df_control_pre = @subset(df, :subject_id .<= 99, :run .== 1, :session .== "bhb")
    df_control_post = @subset(df, :subject_id .<= 99, :run .== 2, :session .== "bhb")
    df_bipolar_pre = @subset(df, :subject_id .> 99, :run .== 1, :session .== "bhb")
    df_bipolar_post = @subset(df, :subject_id .> 99, :run .== 2, :session .== "bhb")

    scatter!(ax[1], fill(1, nrow(df_control_pre)), df_control_post.σ_inf .- df_control_pre.σ_inf; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.σ_inf .- df_bipolar_pre.σ_inf; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control_pre)), df_control_post.β .- df_control_pre.β; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.β .- df_bipolar_pre.β; color=colormap[2])

    colgap!(f.layout, 40)
    xlims!.(ax, 0.8, 2.2)
    ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Dot Category Match model parameter differences",
        fontsize = 30, color = (:black, 0.6))

    if save
        save(string("task1_model_param_diff", ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_faces_model(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors

    fontsize_label = 18
    labels = ["Control", "Bipolar"]

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Initial bias",
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

    df_control = @subset(df, :subject_id .<= 99, :run .== 1)
    df_bipolar = @subset(df, :subject_id .> 99, :run .== 1)

    scatter!(ax[1], fill(1, nrow(df_control)), df_control.z; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar)), df_bipolar.z; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control)), df_control.α; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar)), df_bipolar.α; color=colormap[2])

    scatter!(ax[3], fill(1, nrow(df_control)), df_control.τ; color=colormap[1])
    scatter!(ax[3], fill(2, nrow(df_bipolar)), df_bipolar.τ; color=colormap[2])

    scatter!(ax[4], fill(1, nrow(df_control)), df_control.drift_intercept; color=colormap[1])
    scatter!(ax[4], fill(2, nrow(df_bipolar)), df_bipolar.drift_intercept; color=colormap[2])

    scatter!(ax[5], fill(1, nrow(df_control)), df_control.drift_angry; color=colormap[1])
    scatter!(ax[5], fill(2, nrow(df_bipolar)), df_bipolar.drift_angry; color=colormap[2])

    scatter!(ax[6], fill(1, nrow(df_control)), df_control.drift_neutral; color=colormap[1])
    scatter!(ax[6], fill(2, nrow(df_bipolar)), df_bipolar.drift_neutral; color=colormap[2])

    colgap!(f.layout, 40) 
    xlims!.(ax, 0.8, 2.2)
    #ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Faces Match model parameters",
        fontsize = 26, color = (:black, 0.6))

    if save
        save(string("task3_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end


function figure_faces_model_param_diff(df; save=false)
    colormap = ColorSchemes.seaborn_bright.colors

    fontsize_label = 18
    labels = ["Control", "Bipolar"]

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = [
            Axis(
                f[1, 1], 
                title = "",
                xlabel = "",
                xticks = ([1,2], labels),
                ylabel = "Initial bias",
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

    df_control_pre = @subset(df, :subject_id .<= 99, :run .== 1, :session .== "bhb")
    df_control_post = @subset(df, :subject_id .<= 99, :run .== 2, :session .== "bhb")
    df_bipolar_pre = @subset(df, :subject_id .> 99, :run .== 1, :session .== "bhb")
    df_bipolar_post = @subset(df, :subject_id .> 99, :run .== 2, :session .== "bhb")

    dz_control = df_control_post.z .- df_control_pre.z
    dz_bipolar = df_bipolar_post.z .- df_bipolar_pre.z

    t = EqualVarianceTTest(dz_control, dz_bipolar)
    @show t
    @show pvalue(t)

    scatter!(ax[1], fill(1, nrow(df_control_pre)), df_control_post.z .- df_control_pre.z; color=colormap[1])
    scatter!(ax[1], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.z .- df_bipolar_pre.z; color=colormap[2])

    scatter!(ax[2], fill(1, nrow(df_control_pre)), df_control_post.α .- df_control_pre.α; color=colormap[1])
    scatter!(ax[2], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.α .- df_bipolar_pre.α; color=colormap[2])

    scatter!(ax[3], fill(1, nrow(df_control_pre)), df_control_post.τ .- df_control_pre.τ; color=colormap[1])
    scatter!(ax[3], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.τ .- df_bipolar_pre.τ; color=colormap[2])

    scatter!(ax[4], fill(1, nrow(df_control_pre)), df_control_post.drift_intercept .- df_control_pre.drift_intercept; color=colormap[1])
    scatter!(ax[4], fill(2, nrow(df_bipolar_post)), df_bipolar_post.drift_intercept .- df_bipolar_pre.drift_intercept; color=colormap[2])

    scatter!(ax[5], fill(1, nrow(df_control_pre)), df_control_post.drift_angry .- df_control_pre.drift_angry; color=colormap[1])
    scatter!(ax[5], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.drift_angry .- df_bipolar_pre.drift_angry; color=colormap[2])

    scatter!(ax[6], fill(1, nrow(df_control_pre)), df_control_post.drift_neutral .- df_control_pre.drift_neutral; color=colormap[1])
    scatter!(ax[6], fill(2, nrow(df_bipolar_pre)), df_bipolar_post.drift_neutral .- df_bipolar_pre.drift_neutral; color=colormap[2])

    colgap!(f.layout, 40) 
    xlims!.(ax, 0.8, 2.2)
    #ylims!(ax[2], -1.5, 5)

    supertitle = f[0, :] = Label(f, "Faces Match model parameters",
        fontsize = 26, color = (:black, 0.6))

    if save
        save(string("task3_model_params", ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_psychophysics_CM!(ax::Axis, gdf::GroupedDataFrame, edges; color)
    N_subjects = length(gdf)

    P_right_choices = mapreduce(hcat, enumerate(gdf)) do (i, df)
        C = get_choices(df)
        z_left = get_loglikelihood_choice(df, 1)
        z_right = get_loglikelihood_choice(df, 2)

        Delta_llhood = z_right .- z_left

        P_right_choices_subject = map(eachindex(edges[1:end-1])) do i
            idx = findall(Delta_llhood) do Dl
                (Dl >= edges[i]) && (Dl < edges[i+1])
            end
    
            sum(C[idx]) / length(C[idx])
        end

        P_right_choices_subject
    end
    
    μ_P = vec(mean(P_right_choices; dims=2))
    sem_P = vec(std(P_right_choices; dims=2) ./ sqrt(N_subjects))
    
    scatter!(ax, edges[1:end-1], μ_P; color)
    lines!(ax, edges[1:end-1], μ_P; color)
    errorbars!(ax, edges[1:end-1], μ_P, sem_P; color) 
end

function figure_psychophysics_CM(df::DataFrame; N_points=10, name="", save_fig=false)
    colormap = ColorSchemes.seaborn_bright.colors

    z_left = get_loglikelihood_choice(df, 1)
    z_right = get_loglikelihood_choice(df, 2)
    Delta_llhood = z_right .- z_left

    edges = range(minimum(Delta_llhood), maximum(Delta_llhood); length=(N_points + 1))
    
    f = Figure(;size = (1280, 720), fontsize=26)
    ax = Axis(f[1,1], xlabel = "Likelihood difference (category 2 - category 1)", ylabel = "Probability of category 2")

    gdf = groupby(df, :subject_id)
    plot_psychophysics_CM!(ax, gdf, edges; color=colormap[1])
    
    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_regressor(t_regress, val_regress; pulse_width=1, regressor_name="", name="", save_fig=false)
    ts = 0:pulse_width:(maximum(t_regress) + 4*pulse_width)

    val_padded = zeros(length(ts))
    for (i, t) in enumerate(t_regress)
        idx = argmin(abs.(ts .- t))
        val_padded[idx] = val_regress[i]
    end

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = Axis(f[1,1], xlabel = "Time [sec]", ylabel = regressor_name)

    stairs!(ax, ts, val_padded)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end


function figure_hrf_regressor(t_regress, val_regress; pulse_width=1, regressor_name="", name="", save_fig=false)
    
    ts = 0:pulse_width:(maximum(t_regress) + 4*pulse_width)
    val_padded = zeros(length(ts))
    for (i, t) in enumerate(t_regress)
        idx = argmin(abs.(ts .- t))
        val_padded[idx] = val_regress[i]
    end
    val_padded ./= maximum(val_regress)

    val_hrf = spm_hrf_convolve(val_padded, 1)
    val_hrf ./= maximum(val_hrf)

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = Axis(f[1,1], xlabel = "Time [sec]", ylabel = string(regressor_name, " (HRF)"))

    lines!(ax, ts, val_hrf)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function figure_combined_regressor(t_regress, val_regress; pulse_width=1, regressor_name="", name="", save_fig=false)
    colormap = ColorSchemes.seaborn_bright.colors

    ts = 0:pulse_width:(maximum(t_regress) + 4*pulse_width)
    val_padded = zeros(length(ts))
    for (i, t) in enumerate(t_regress)
        idx = argmin(abs.(ts .- t))
        val_padded[idx] = val_regress[i]
    end
    val_padded ./= maximum(val_regress)

    val_hrf = spm_hrf_convolve(val_padded, 1)
    val_hrf ./= maximum(val_hrf)

    f = Figure(;size = (1280, 720), fontsize=26)
    ax = Axis(f[1,1], xlabel = "Time [sec]", ylabel = regressor_name)

    stairs!(ax, ts, val_padded; color=colormap[1], label="Raw")
    lines!(ax, ts, val_hrf; color=colormap[2], label="HRF")

    f[1, 2] = Legend(f, ax; framevisible = false)

    if save_fig
        save(string(name, ".png"), f, pt_per_unit=1)
    end

    f
end

function plot_prob_choice(L, β, σ; save=false)
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

    if save
        save("example_prob_choice.png", f, pt_per_unit=1)
    end

    f
end
