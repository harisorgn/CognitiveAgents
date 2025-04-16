kernel(t1, t2; β=1) = β / (t1 - t2)


function gauss_mask(N, σ)
    P = zeros((N^2, N^2))
    c = 1
    for i in Base.OneTo(N)
        for j in Base.OneTo(N)
            m = zeros(N,N)
            for di in Base.OneTo(N)
                for dj in Base.OneTo(N)
                    dsq = (i-di)^2 + (j - dj)^2
                    #m[di, dj] = exp(-dsq / (2*σ^2)) / (2*pi*σ^2)
                    m[di, dj] = exp(-dsq / σ^2)
                end
            end
            P[c, :] = vec(m)
            c += 1
        end
    end

    return P
end

function convolution(IMG; σ=1)
    N = size(IMG)[1]
    M = gauss_mask(N, σ)
    C = M * vec(IMG)

    return C
end

get_corrects(df) = map(x -> occursin("True", x) ? true : false, df.correct) 

get_choices(df) = occursin.("right", df.response)

get_choicesp1(df) = Int.(occursin.("right", df.response)) .+ 1

get_correct_categories(df::DataFrame) = Int.(occursin.("right", df.correct_response))

get_corrects(df::DataFrame) = return eltype(df.correct) <: Bool ? df.correct : parse.(Bool, df.correct)

function get_response_times(df)
    RT = df.response_time
    r = Vector{Union{Float64, Missing}}(undef, length(RT))

    idx_missing = ismissing.(RT) .| (RT .== "None") 

    if any(idx_missing)
        r[idx_missing] .= missing
        r[.!(idx_missing)] .= parse.(Float64, RT[.!(idx_missing)])
    else
        r .= parse.(Float64, RT)
    end
    
    
    return r
end

function get_stimuli(df; σ_conv=1, grid_sz=(50,50))
    X = mapreduce(hcat, eachrow(df)) do r
        set = parse(Int, r.set)
        cat = parse(Int, r.category)
        ID = parse(Int, r.stimulus_ID)
        pack = r.version
        IMG = imresize(load_image(pack, set, cat, ID), grid_sz)

        C = convolution(IMG; σ = σ_conv)
        Float64.(Gray.(C))
        
    end
    X ./= maximum(X)

    return X
end

function get_loglikelihood_dots(df)
    map(eachrow(df)) do r
        set = parse(Int, r.set)
        cat = parse(Int, r.category)
        ID = parse(Int, r.stimulus_ID)
        pack = parse(Int, last(split(r.version, '_')))
        llhoods = load_loglikelihood(pack, set, cat, ID)  
        #llhoods ./ abs.(minimum(llhoods; dims=1))
    end
end

function get_loglikelihood_choice(df, choice_idx)
    L = get_loglikelihood_dots(df)
    RD = get_response_dots(df)

    z = map(enumerate(L)) do (t, loglikelihoods)
        sum(loglikelihoods[1:RD[t], choice_idx])
    end

    return z
end

function get_response_dots(df; inter_dot_interval = 0.55)
    RT = get_response_times(df)
    dots = div.(RT, inter_dot_interval, RoundDown)

    return Int.(dots)
end

get_image_IDs(df) = parse.(Int, df.image_response)

load_image(pack, set, category, ID) = load("./stimuli/$(pack)/set_$(set)/cat_$(category)/ex_$(category)_$(ID).png")

function load_loglikelihood(pack, set, category, ID)
    readdlm("./stimuli/SNR_easy_$(pack)/set_$(set)/cat_$(category)/ex_$(category)_$(ID)_loglikelihood.txt", ',')
end