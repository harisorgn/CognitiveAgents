using CognitiveAgents
using Serialization
using Turing

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

session = "glc"
run = 1

#df_fit = df[(df.subject_id .== 9) .& (df.run .== run) .& (df.session .== session), :]
df_fit = df[(df.subject_id .== 9), :]

using CognitiveAgents: get_loglikelihood_dots, get_choices, get_response_dots, discrete_evidence

L = get_loglikelihood_dots(df_fit)
C = get_choices(df_fit)
RD = get_response_dots(df_fit)

model = discrete_evidence(RD, L, C)
sig, P_lapse, choices = model()

model = discrete_evidence(RD, L, choices)
chain = sample(model, Emcee(40), 2_000, progress=false)
chain_prior = sample(model, Prior(), 2_000, progress=false)

res = fit_bayes(df_fit)

#alg = NLopt.GN_CRS2_LM()
for ID in IDs
    df_fit = df[(df.subject_id .== ID) .& (df.run .== run) .& (df.session .== session), :]
    res = fit_model(df_fit, alg)

    filename = "CL_res_subj-$(ID)_ses-$(session)_run-$(run).jls"
    serialize(filename, res)
end

σ_conv = 10
grid_sz = (50,50)
