
chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]

omissions(df) = count(ismissing.(df[!, :response]))

function read_data_js(files, cols)
    df = DataFrame()
    for f in files
        df_subj = CSV.read(f, DataFrame)
        if !("subject_id" in names(df_subj)) || all(ismissing.(df_subj.subject_id))
            df_subj.subject_id .= rand(1001:9999)
        end
        df_subj.bonus[ismissing.(df_subj.bonus)] .= maximum(skipmissing(df_subj.bonus))
        df_subj.task[ismissing.(df_subj.task)] .= ""
        df_subj.response[ismissing.(df_subj.response)] .= ""
        df_subj.correct[ismissing.(df_subj.correct)] .= 0

        bonus = df_subj[findlast(x -> !ismissing(x), df_subj.bonus), :bonus]
        df_subj.bonus .= bonus

        df_subj = df_subj[df_subj.task .== "response", cols]
        
        df_subj.subject_id = string.(df_subj.subject_id)
        if (omissions(df_subj) / nrow(df_subj) <= 0.1)
            append!(df, df_subj)
        end
    end

    return df
end

function read_data_psychopy(files, cols)
    df = DataFrame()
    for f in files
        df_subj = CSV.read(f, DataFrame, types=String)
        if !("subject_id" in names(df_subj)) || all(ismissing.(df_subj.subject_id))
            df_subj.subject_id .= string(rand(1001:9999))
        end
        df_subj.response[ismissing.(df_subj.response)] .= ""
        df_subj.correct[ismissing.(df_subj.correct)] .= "0"

        df_subj.response_time[ismissing.(df_subj.response_time)] .= "4.0"
        
        if (omissions(df_subj) / nrow(df_subj) <= 0.1)
            append!(df, df_subj[:, cols])
        end
    end

    return df
end

function read_data_bipolar(files, cols; include_omissions=false)
    df = DataFrame()
    
    for f in files
        fsplt = split((split(f, '.'))[end-1], '_')
        sub_idx = findfirst(fi -> occursin("sub-",fi), fsplt)
        subject_ID = parse(Int, last(split(fsplt[sub_idx], '-')))

        session_idx = findfirst(fi -> occursin("ses-",fi), fsplt)
        session = last(split(fsplt[session_idx], '-'))

        run_idx = findfirst(fi -> occursin("run-",fi), fsplt)
        run = last(split(fsplt[run_idx], '-'))

        df_subj = CSV.read(f, DataFrame, types=String)

        df_subj.trial_index = Base.OneTo(nrow(df_subj))
        df_subj.subject_id .= subject_ID
        df_subj.session .= session
        df_subj.run .= parse(Int, run)

        if "pack" in names(df_subj)
            extra_cols = [:trial_index, :run, :session, :pack]
            df_subj.pack .= df_subj.version
        else
            extra_cols = [:trial_index, :run, :session]
        end

        df_subj.response[ismissing.(df_subj.response)] .= ""
        df_subj.correct[ismissing.(df_subj.correct)] .= "0"

        if "bonus" in names(df_subj)
            df_subj.bonus[ismissing.(df_subj.bonus)] .= "0.0"
        end

        if (omissions(df_subj) / nrow(df_subj) <= 0.1)
            if include_omissions
                append!(df, df_subj[:, vcat(cols, extra_cols)])
            else    
                idx_include = .!(ismissing.(df_subj.response_time) .| (df_subj.response_time .== "None"))  
                append!(df, df_subj[idx_include, vcat(cols, extra_cols)])
            end
        end
    end

    df.correct .= parse.(Int, df.correct)
    return df
end

function read_aggressiveness(df::DataFrame; normalize=true)
    df_agr = CSV.read("./data/aggressiveness.csv", DataFrame)

    if normalize
        return df_agr[df.trial_index, [:score]] .- mean(df_agr.score)
    else
        return df_agr[df.trial_index, [:score]]
    end
end

function subject_bonus(df)
    for ID in unique(df[!, :subject_id])
        println("$(ID) => $(maximum(df[df.subject_id .== ID, :bonus]))")
    end
end
