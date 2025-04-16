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

task = "task2"
dir = joinpath("./data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)
filter!(r -> (r.subject_id <= 99) .& (r.run == 1) .& (r.phase == "test"), df)


run = 1
session = "bhb"
IDs = unique(df.subject_id)
alg = Optim.IPNewton()
for ID in [104]
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    res = fit_CM(df_fit, alg)
    
    serialize("CM_model_sub-$(ID)_ses-$(session)_run-$(run).csv", res)

    results_to_regressors(res, df)
end

#figure_CM_psychophysics(df_fit, res; N_points=10, save_fig=true)
