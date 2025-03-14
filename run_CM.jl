using CognitiveAgents
using Serialization
using OptimizationOptimJL

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

task = "task2"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)
filter!(r -> r.phase == "test", df)

IDs = unique(df.subject_id)

df_fit = df[(df.subject_id .== 9) .& (df.run .== 1) .& (df.session .== "bhb"), :]

alg = Optim.IPNewton()
res = fit_CM(df_fit, alg)

figure_CM_psychophysics(df_fit, res; N_points=10)
