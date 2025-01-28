module CognitiveAgents

using Distributions

using CommonRLInterface: AbstractEnv

using Optimization
using Optim

using CSV
using DataFrames

using LogExpFunctions: logsumexp, logistic

using NNlib

using Images: load, imresize, Gray

using UnPack
using Serialization

using CairoMakie
using ColorSchemes

include("utils.jl")
include("read.jl")
include("EM_learning.jl")
include("discrete_evidence.jl")
include("CategoryLearnEnv.jl")
include("plot.jl")

export read_data_bipolar, read_data_psychopy, read_data_js
export fit_model
export CLResult, CMResult
export EMAgent

end 
