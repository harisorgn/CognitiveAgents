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

function spm_Gpdf(x, h, l)

    # x - Gamma-variate   (Gamma has range [0,Inf) )
    # h - Shape parameter (h>0)
    # l - Scale parameter (l>0)
    # f - PDF of Gamma-distribution with shape & scale parameters h & l

    # Need to set the first value to 0 and then the rest to the function, so this extrat step and indexing takes care of that
    f = [0.0; exp.((h-1) .* log.(x[2:end]) .+ h.*log(l) .- l.*x[2:end] .- loggamma(h))]
end

function spm_hrf(RT; p=[6, 16, 1, 1, 6, 0, 32], T=16)
    # RT   - scan repeat time
    # p    - parameters of the response function (two Gamma functions)
    #
    #                                                     defaults
    #                                                    {seconds}
    #  p(1) - delay of response (relative to onset)          6
    #  p(2) - delay of undershoot (relative to onset)       16
    #  p(3) - dispersion of response                         1
    #  p(4) - dispersion of undershoot                       1
    #  p(5) - ratio of response to undershoot                6
    #  p(6) - onset {seconds}                                0
    #  p(7) - length of kernel {seconds}                    32
    #
    # T    - microtime resolution [Default: 16]
    #
    # hrf  - haemodynamic response function
    # p    - parameters of the response function

    # Modelled hemodynamic response function - SPM double gamma
    dt = RT/T
    u = collect(0:ceil(p[7]/dt) - p[6]/dt)
    hrf = spm_Gpdf(u, p[1]/p[3], dt/p[3]) - spm_Gpdf(u, p[2]/p[4], dt/p[4])/p[5]
    hrf = hrf[collect(0:Int(floor(p[7]/RT))).*T .+ 1]
    hrf = hrf/sum(hrf)
    return hrf
end

function spm_hrf_convolve(stimulus, RT=0.8)
    hrf = spm_hrf(RT)
    final = conv(hrf, stimulus)
    return final[1:length(stimulus)]
end
