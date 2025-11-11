using CognitiveAgents
using Serialization
using OptimizationNLopt
using DataFrames
using DataFramesMeta
using UMAP
using CairoMakie
using ColorSchemes

fs = readdir("./results/category_learn/"; join=true)
res = deserialize.(fs)
df = results_to_dataframe(res)


df_baseline = @subset(df, :run .== 1)
figure_CL_model(df_baseline; save_fig=true, name="cl_params_baseline")

df_glc = @subset(df, :run .== 2, :session .== "glc")
figure_CL_model(df_glc; save_fig=true, name="cl_params_glc")

df_bhb = @subset(df, :run .== 2, :session .== "bhb")
figure_CL_model(df_bhb; save_fig=true, name="cl_params_bhb")



X = transpose(Matrix(df[!, [:η, :ηₓ, :α, :β, :σ²]]))

idxs_ctrl_base = (df.subject_id .<= 99) .& (df.run .== 1)
idxs_bp_base = (df.subject_id .> 99) .& (df.run .== 1)
idxs_ctrl_glc = (df.subject_id .<= 99) .& (df.run .== 2) .& (df.session .== "glc")
idxs_bp_glc = (df.subject_id .> 99) .& (df.run .== 2) .& (df.session .== "glc")
idxs_ctrl_bhb = (df.subject_id .<= 99) .& (df.run .== 2) .& (df.session .== "bhb")
idxs_bp_bhb = (df.subject_id .> 99) .& (df.run .== 2) .& (df.session .== "bhb")

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
