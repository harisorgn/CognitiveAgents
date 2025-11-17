using CognitiveAgents
using Serialization
using OptimizationNLopt
using DataFrames
using DataFramesMeta
using UMAP
using CairoMakie
using ColorSchemes
using CSV

task = "task1"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))
filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)
df_data = read_data_bipolar(files, cols)

df_params = CSV.read("./results/category_learn/CL_model_params.csv", DataFrame)

σ_conv = 5
grid_sz = (50,50)

z = map(eachrow(df_params)) do row
    get_categorization_rules(df_data, df_params, row.subject_id, row.session, row.run; σ_conv, grid_sz)
end

df_params.N_rules = maximum.(z)



df_baseline = @subset(df_params, :run .== 1)
figure_CL_model(df_baseline; save_fig=true, name="cl_params_baseline")

df_glc = @subset(df_params, :run .== 2, :session .== "glc")
figure_CL_model(df_glc; save_fig=true, name="cl_params_glc")

df_bhb = @subset(df_params, :run .== 2, :session .== "bhb")
figure_CL_model(df_bhb; save_fig=true, name="cl_params_bhb")



df_glc = @subset(df_params, :session .== "glc")
figure_CL_model_param_diff(df_glc; save_fig=true, name="cl_params_diff_glc")

df_bhb = @subset(df_params, :session .== "bhb")
figure_CL_model_param_diff(df_bhb; save_fig=true, name="cl_params_diff_bhb")




X = transpose(Matrix(df_params[!, [:η, :ηₓ, :α, :β, :σ²]]))

idxs_ctrl_base = (df_params.subject_id .<= 99) .& (df_params.run .== 1)
idxs_bp_base = (df_params.subject_id .> 99) .& (df_params.run .== 1)
idxs_ctrl_glc = (df_params.subject_id .<= 99) .& (df_params.run .== 2) .& (df_params.session .== "glc")
idxs_bp_glc = (df_params.subject_id .> 99) .& (df_params.run .== 2) .& (df_params.session .== "glc")
idxs_ctrl_bhb = (df_params.subject_id .<= 99) .& (df_params.run .== 2) .& (df_params.session .== "bhb")
idxs_bp_bhb = (df_params.subject_id .> 99) .& (df_params.run .== 2) .& (df_params.session .== "bhb")

emb = umap(X, 2; n_neighbors=10, min_dist=0.01)

colormap = ColorSchemes.seaborn_bright.colors;

fig = Figure()
ax = Axis(fig[1,1], xlabel="UMAP 1", ylabel="UMAP 2")

scatter!(ax, emb[:, idxs_ctrl_base]; color=colormap[1], label="Control Baseline")
scatter!(ax, emb[:, idxs_bp_base]; color=colormap[2], label="Bipolar Baseline")
scatter!(ax, emb[:, idxs_ctrl_glc]; color=colormap[3], label="Control GLC")
scatter!(ax, emb[:, idxs_bp_glc]; color=colormap[4], label="Bipolar GLC")
scatter!(ax, emb[:, idxs_ctrl_bhb]; color=colormap[5], label="Control BHB")
scatter!(ax, emb[:, idxs_bp_bhb]; color=colormap[6], label="Bipolar BHB")

fig[1, 2] = Legend(fig, ax; framevisible = false)

fig
