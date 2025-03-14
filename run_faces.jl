using CognitiveAgents
using Serialization
using OptimizationOptimJL

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

df_fit = df[(df.subject_id .== 3) .& (df.run .== 1) .& (df.session .== "bhb"), :]

alg = Optim.IPNewton()
res = fit_faces(df_fit, alg)

figure_faces_psychophysics(df_fit, res)

figure_faces_RT(df_fit, res)