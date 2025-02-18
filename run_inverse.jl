using CognitiveAgents
using Serialization
using OptimizationNLopt

function fit_batch(IDs, session, run, algs, grid_sz, σ_conv; kwargs...)
    for ID in IDs
        df_fit = df[(df.subject_id .== ID) .& (df.run .== run) .& (df.session .== session), :]
    
        filename = "CL_res_subj-$(ID)_ses-$(session)_run-$(run)"
        
        alg = algs[1]
        res = fit_model(df_fit, alg; grid_sz, σ_conv, kwargs...)

        for alg in algs[2:end]
            p0 = res.sol.u
            res = fit_model(df_fit, alg, p0; grid_sz, σ_conv, kwargs...)
        end

        serialize(string("./results/", filename, ".jls"), res)
    end  
end

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

#IDs = unique(df.subject_id)
IDs = [9, 36, 38, 40, 41, 43]
alg = NLopt.GN_MLSL_LDS()

session = "glc"
run = 1
for ID in IDs
    df_fit = df[(df.subject_id .== ID) .& (df.run .== run) .& (df.session .== session), :]
    res = fit_model(df, alg)

    filename = "CL_res_subj-$(ID)_ses-$(session)_run-$(run).jls"
    serialize(filename, res)
end

σ_conv = 10
grid_sz = (50,50)
