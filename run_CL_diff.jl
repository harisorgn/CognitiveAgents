using CognitiveAgents
using Serialization
using OptimizationOptimJL
using DataFramesMeta

cols = [
    :subject_id,
    :stimulus_ID,
    :category,
    :set,
    :response,
    :correct,
    :correct_response,
    :response_time,
    :stim_presentation_time,
    :phase,
    :version
]

task = "task1"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)

IDs = unique(df.subject_id)

σ_conv = 5
grid_sz = (50,50)
alg = Optim.IPNewton()

run = 1
session = "glc"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_CL(df_fit, alg; grid_sz, σ_conv)
        
        serialize("./results/category_learn_2/CL_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end

session = "bhb"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_CL(df_fit, alg; grid_sz, σ_conv)
        
        serialize("./results/category_learn_2/CL_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end

fs = filter(f -> occursin("CL_model", f) && occursin("run-1", f), readdir("."))
res = CLResult[]
for f in fs
    @show f
    push!(res, deserialize(f))
end

dfr = results_to_dataframe(res)

figure_CL_model(dfr; save_fig=true)