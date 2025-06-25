using CognitiveAgents
using Serialization
using OptimizationOptimJL
using DataFramesMeta
using Turing
using ForwardDiff, Preferences

set_preferences!(ForwardDiff, "nansafe_mode" => true)

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

choices = Vector{Int}[]
corrects = Vector{Int}[]
group_index = Int[]

for session in ["glc", "bhb"]
    for ID in IDs
        df_fit = @subset(df, :subject_id .== ID, :run .== 1, :session .== session)
        if !isempty(df_fit)
            push!(choices, get_choices(df_fit))
            push!(corrects, get_correct_categories(df_fit))
            #push!(S, get_stimuli(df_fit; grid_sz, σ_conv))
            push!(group_index, ID <= 99 ? 1 : 2)
        end
    end
end
S = deserialize("stimuli.jls")

setprogress!(false)

model = hierarchical_SLP(choices, group_index, S, corrects)

chain = sample(
            model, 
            NUTS(0.85), 
            MCMCThreads(),
            2_000, 
            4
)
