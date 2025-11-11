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

for run in [1, 2]
    for session in ["glc", "bhb"]
        for ID in IDs
            df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

            if !isempty(df_fit)
                res = fit_CL(df_fit, NLopt.LN_BOBYQA(); reltol=1e-8, abstol=1e-8)
                serialize("CL_sub-$(ID)_ses-$(session)_run-$(run).jls", res)
            end
        end
    end
end
