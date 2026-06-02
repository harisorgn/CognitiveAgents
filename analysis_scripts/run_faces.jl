using CognitiveAgents
using Serialization
using DataFramesMeta
using CSV

cols = [
    :subject_id,
    :image_response,
    :response,
    :correct,
    :response_time
]

task = "task3"
dir = joinpath(@__DIR__, "../data", "bipolar")
files = mapreduce(x -> readdir(x; join=true), vcat, readdir(dir; join=true))

filter!(f -> (last(split(f,'.')) == "csv") && (occursin(task, f)), files)

df = read_data_bipolar(files, cols)

IDs = unique(df.subject_id)

run = 1
session = "bhb"
res = FacesResult[]
for ID in IDs[1:5]
    df_fit = @subset(df, :subject_id .== ID, :run .== run, :session .== session)

    if !isempty(df_fit)
        res_subj = fit_faces(df_fit)
        push!(res, res_subj)
    end
end

df_res = results_to_dataframe(res)
CSV.write("faces_model_params_$(session)_run_$(run).csv", df_res)

faces_results_to_regressors(df_res, df)

figure_RT_faces(df)

figure_psychophysics_faces(df)

