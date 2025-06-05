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
alg = Optim.IPNewton()

run = 2
session = "bhb"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_faces(df_fit, alg)
        serialize("faces_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end

session = "glc"
for ID in IDs
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res = fit_faces(df_fit, alg)
        serialize("faces_model_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
    end
end

dir = joinpath("./results", "faces_match")
files = readdir(dir; join=true)
res = deserialize.(files)
df = results_to_dataframe(res)

figure_faces_model(df)

figure_faces_model_param_diff(df)
