using CognitiveAgents
using Serialization
using OptimizationOptimJL
using DataFramesMeta

cols = [
    :subject_id,
    :image_response,
    :response,
    :correct,
    :response_time
]

task = "task3"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)

IDs = unique(df.subject_id)

run = 1
session = "glc"
IDs = unique(df.subject_id)
alg = Optim.IPNewton()
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    res = fit_faces(df_fit, alg)
    
    serialize("faces_model_sub-$(ID)_ses-$(session)_run-$(run).csv", res)

    results_to_regressors(res, df)
end

#figure_faces_psychophysics(df_fit, res; save_fig=true)

#figure_faces_RT(df_fit, res; save_fig=true)