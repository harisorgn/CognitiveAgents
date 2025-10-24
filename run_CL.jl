using CognitiveAgents
using Serialization
using OptimizationNLopt
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

df_fit = @subset(df, :subject_id .== IDs[1], :run .== 1, :session .== "glc")
S = get_stimuli(df_fit; σ_conv, grid_sz)
choices = get_choices(df_fit)
corrects = get_correct_categories(df_fit)

ag = initialise_agent(S, [0.1, 0.1])

loglikelihood(ag, S, choices, corrects)


alg = NLopt.LN_BOBYQA()

r = fit_EM(df_fit, alg)