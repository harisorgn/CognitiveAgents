mutable struct CategoryLearnEnv <: AbstractEnv
    const S::Matrix{Float64}
    const C::Vector{Int}
    const N_trials::Int
    current_trial::Int
    
    function CategoryLearnEnv(df::DataFrame)
        S = transpose(Matrix(df[!, Not(:category)]))
        C = df[!, :category]

        new(S, C, length(C), 1)
    end

    function CategoryLearnEnv(S::AbstractMatrix, C::AbstractVector)
        new(S, C, length(C), 1)
    end
end

function CommonRLInterface.reset!(env::CategoryLearnEnv)
    env.current_trial = 1
end

CommonRLInterface.actions(::CategoryLearnEnv) = (1, 2)

CommonRLInterface.observe(env::CategoryLearnEnv) = env.S[:, env.current_trial]

CommonRLInterface.terminated(env::CategoryLearnEnv) = env.current_trial > env.N_trials

# For now acting on the environment only returns a reward
# and does not increment! the environment forward as `CommonRLInterface` dictates.
# This is done because of N_loops in EMAgent, 
# where we don't want to increment! the environment on every loop. 
CommonRLInterface.act!(env::CategoryLearnEnv, action) = env.C[env.current_trial]

increment!(env::CategoryLearnEnv) = env.current_trial += 1

current_trial(env::CategoryLearnEnv) = env.current_trial