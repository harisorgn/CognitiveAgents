using CognitiveAgents
using DataFrames
using OptimizationNLopt
using OptimizationBBO

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

df_subj = subset(df, :subject_id => id -> id .== IDs[1], :run => r -> r.==1)

S = get_stimuli(df_subj; grid_sz, σ_conv)
corrects = get_correct_categories(df_subj)
env = CategoryLearnEnv(S, corrects)

η = 0.05
ηₓ = 0.05
α = 1
agent = EMAgent(S; η, ηₓ, α, σ²=2)

choices = run_task!(agent, env)

#alg = NLopt.GN_MLSL_LDS()
alg = GridSearch(0.01)
res = fit_model(S, choices, corrects, alg; grid_sz, σ_conv)
