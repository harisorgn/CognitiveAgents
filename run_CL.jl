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

task = "task1"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)
filter!(r -> r.phase == "test", df)

IDs = unique(df.subject_id)

σ_conv = 5
grid_sz = (50,50)

df_fit = df[(df.subject_id .== 1) .& (df.run .== 1) .& (df.session .== "bhb"), :]

alg = Optim.IPNewton()
res = fit_CL(df_fit, alg; grid_sz, σ_conv)

#d = run_CL_task(df_fit, res)

figure_subject_accuracy(df_fit, res)
