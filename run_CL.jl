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

run = 2
session = "glc"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_CL(df_fit, alg; grid_sz, σ_conv)
        
        serialize("CL_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end

session = "bhb"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_CL(df_fit, alg; grid_sz, σ_conv)
        
        serialize("CL_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end


dir = joinpath("./results", "category_learn")
files = readdir(dir; join=true)
res = deserialize.(files)
df = results_to_dataframe(res)

figure_CL_model(df)

figure_CL_model_param_diff(df)

