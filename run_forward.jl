using CognitiveAgents
using DataFrames
using OptimizationNLopt

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

σ_conv = 10
grid_sz = (50,50)

df_subj = subset(df, :subject_id => id -> id .== IDs[1], :run => r -> r.==1, :session => s -> s.== "bhb")

C = get_correct_categories(df_subj)
S = get_stimuli(df_subj; grid_sz, σ_conv)
env = CategoryLearnEnv(S, C)

η = 0.2
ηₓ = 0.15
α = 2
agent = EMAgent(S; η, ηₓ, α)

choices = run_task!(agent, env)

alg = NLopt.GN_MLSL_LDS()
res = fit_model(S, choices, alg; grid_sz, σ_conv)
