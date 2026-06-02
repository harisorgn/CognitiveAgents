module CognitiveAgents

using Distributions

using JuMP
using MadNLP

using DelimitedFiles: readdlm
using CSV
using DataFrames
using DataFramesMeta

using SequentialSamplingModels: DDM

using LogExpFunctions: logsumexp, logistic, logit
using NNlib: softmax

using Statistics: mean, std
using StatsBase: ecdf, Histogram, fit

using LinearAlgebra: normalize

using Base.Iterators: partition

using Images: load, imresize, Gray

using DSP
using SpecialFunctions

using HypothesisTests: EqualVarianceTTest, pvalue

using UnPack

using CairoMakie
using ColorSchemes

include("utils.jl")
include("read.jl")
include("category_learn.jl")
include("category_match.jl")
include("faces_match.jl")
include("plot.jl")

export read_data_bipolar, read_aggressiveness, read_data_psychopy, read_data_js
export get_choices, get_correct_categories, get_stimuli, get_response_times
export CLResult, CMResult, FacesResult, EMAgent, CategoryLearnEnv
export fit_CL, fit_CM, fit_faces
export CL_results_to_regressors, CM_results_to_regressors, faces_results_to_regressors, results_to_dataframe, spm_hrf_convolve
export loglikelihood, run_trial!, run_task!, initialise_agent, get_categorization_rules
export negative_loglikelihood
export figure_psychophysics_CM, figure_psychophysics_faces, figure_RT, figure_RT_faces, figure_cumulative_RT, figure_accuracy
export figure_regressor, figure_hrf_regressor, figure_combined_regressor
export figure_CL_model, figure_CM_model, figure_faces_model
export figure_CL_model_param_diff, figure_CM_model_param_diff, figure_faces_model_param_diff

end 
