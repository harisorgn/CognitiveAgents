module CognitiveAgents

using Distributions

using CommonRLInterface
import CommonRLInterface: reset!, actions, observe, terminated, act!

using Turing

using Optimization
using Optim

using DelimitedFiles: readdlm
using CSV
using DataFrames

using LogExpFunctions: logsumexp, logistic
using NNlib: softmax

using Statistics: mean, std

using Base.Iterators: partition

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
export GridSearch, CLResult, CMResult, EMAgent, CategoryLearnEnv
export initialise_agent, objective, fit_model, fit_CM_bayes, fit_CM_optimization, run_trial!, run_task!
export plot_subject_accuracy, plot_group_accuracy, plot_group_accuracy_per_set, plot_rt
export figure_CM_psychometric

end 
