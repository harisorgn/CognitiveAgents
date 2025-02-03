module CognitiveAgents

using Distributions

using CommonRLInterface
import CommonRLInterface: reset!, actions, observe, terminated, act!

using Optimization
using Optim

using CSV
using DataFrames

using LogExpFunctions: logsumexp, logistic

using Images: load, imresize, Gray

using UnPack
using Serialization

using CairoMakie
using ColorSchemes

include("utils.jl")
include("read.jl")
include("CategoryLearnEnv.jl")
include("EMAgent.jl")
include("discrete_evidence.jl")
include("plot.jl")

export read_data_bipolar, read_data_psychopy, read_data_js 
export get_choices, get_correct_categories, get_stimuli, get_response_times
export CLResult, CMResult, EMAgent, CategoryLearnEnv
export initialise_agent, fit_model, run_trial!, run_task!

end 
