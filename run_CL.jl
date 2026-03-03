using CognitiveAgents
using Serialization
using DataFramesMeta
using CSV

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

run = 2
session = "bhb"

res = CLResult[] 
for ID in IDs
    @show ID
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res_subj = fit_CL(df_fit; σ_conv, grid_sz)
        push!(res, res_subj)
    end
end

#=
df_fit = @subset(df, :subject_id .== IDs[10], :run .== 1, :session .== "glc")

res = fit_CL(df_fit; maxtime = 5000.0, σ_conv, grid_sz)

choices = get_choices(df_fit)
corrects = get_correct_categories(df_fit)
S = get_stimuli(df_fit; grid_sz, σ_conv)

negative_loglikelihood([0.8, 0.95, 5.3], S, choices, corrects)

chain = sample(model, dynamic_nuts, 1_000, progress=true)
=#
