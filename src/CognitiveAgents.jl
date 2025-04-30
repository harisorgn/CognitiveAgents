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
using DataFramesMeta

using SequentialSamplingModels: DDM

using LogExpFunctions: logsumexp, logistic
using NNlib: softmax

using Statistics: mean, std
using StatsBase: ecdf, Histogram, fit

using LinearAlgebra: normalize

using Base.Iterators: partition

using Images: load, imresize, Gray

using DSP
using SpecialFunctions

using UnPack
using Serialization

using CairoMakie
using ColorSchemes

include("utils.jl")
include("read.jl")
include("CategoryLearnEnv.jl")
include("category_learn.jl")
include("category_match.jl")
include("faces_match.jl")
include("plot.jl")

export read_data_bipolar, read_aggressiveness, read_data_psychopy, read_data_js
export get_choices, get_correct_categories, get_stimuli, get_response_times
export GridSearch, CLResult, CMResult, EMAgent, CategoryLearnEnv
export objective, category_match, fit_CL, fit_CM, fit_CM, fit_faces, run_CL_task
export figure_psychophysics_CM, figure_psychophysics_faces, figure_RT, figure_RT_faces, figure_group_accuracy
export figure_regressor, figure_hrf_regressor, figure_combined_regressor
export results_to_regressors, spm_hrf_convolve

end 
